# PyTorch Distributed: Experiences on Accelerating Data Parallel Training

## Abstract

In general, the technique of distributed data parallelism replicates the model on every computational resource to generate gradients independently and then communicates those gradients at each iteration to keep model replicas consistent. Despite the technique's conceptual simplicity, subtle dependencies between computation and communication make it non-trivial to optimize distributed training efficiency. As of v1.5, PyTorch natively provides several techniques to accelerate distributed data parallel, including bucketing gradients, overlapping computation with communication, and skipping gradient synchronization.

This lesson will introduce the techniques and provide practical guidance on how to use them to accelerate distributed data parallel training.

## Review

In the training phase, the model performs three steps: forward to compute the loss, backward to compute gradients, and an optimizer step to update the model parameters. In distributed data parallel training, the forward and backward steps are executed independently on each device, while the optimizer step requires communication to synchronize the gradients across devices. The communication overhead can become a bottleneck, especially when training large models or using many devices. This is a pain point for distributed training, so overcoming the last bit of the performance bottleneck consumes a lot of effort in designing and tuning the model communication. We can summarize the pain points as follows:

- _Mathematical equivalence_: The purpose of data parallelism is to speed up training on large datasets. Applications expect to harvest the same model result as if all training had been performed locally without model replication. This requires mathematical equivalence to local training despite its distributed nature. (How to make sure the results of distributed training are mathematically equivalent to local training, and how to verify the equivalence?)
- _Non-intrusive and interceptive API_: Application development usually starts from local models and then scales out when necessary. To avoid the exorbitant hurdles during the transition, the API must be non-intrusive in application code. On the other hand, the API needs to allow the internal implementation to timely intercept signals to carry out communications and system optimizations. (How to design the GPU communication API, which can be GPU-to-GPU or node-to-node communication, to be efficient is a key challenge.)
- _High Performance_: Data parallel training is subject to subtle dependencies between computations and communications. The design and implementation have to explore the solution space to efficiently convert more resources into higher training throughput. (How to maximize training throughput by overlapping computation and communication and by reducing the communication overhead?)

## System Design

Inside the DDP module, the model is replicated on each device, and the forward and backward passes are executed independently on each device. At the final step of every iteration, we need to bring all replicas to the same state. The DDP module includes two main components: API and Gradient Reduction.

### API

A Python API is provided to help users easily interact with the distributed training system. The API is designed with two main goals: non-intrusive and interceptive, allowing users to write code as if they are writing for a single device while the system handles the complexities of distributed training.

- _Non-intrusive_: The API should not require significant changes to the existing codebase, making it easy to integrate into existing applications.
- _Interceptive_: The API should allow the internal implementation to intercept and handle communication and optimization tasks without requiring explicit intervention from the user.

### Gradient Reduction

#### A naive solution

CONDITION: Initially, the model is replicated on each device; each device computes the forward and backward passes independently.

In a naive solution, the model must finish the backward pass on each device before any communication can occur. This leads to a situation where GPUs are idle while waiting for the others to finish their backward passes. Besides GPUs being idle, they are also idle during the communication phase, which can be a significant bottleneck in distributed training. Because of the idle time, the overall training throughput is reduced, and the training process becomes less efficient. Two main issues arise from this naive solution:

1. Collective communication is performed poorly on small tensors, which will be prominent on large models with many small parameters.
2. Separating gradient computation and synchronization forfeits the opportunity to overlap computation with communication due to the hard boundary in between.

The naive solution is visualized in the following figure:

```text
              Forward          Backward    Wait  Communication
Time   0  1  2  3  4  5  6  7  8  9  10
│────│────│────│────│────│────│────│────│────│────│────│────│
GPU 0   ████████████│████████████████████│░░░░░░░│████████
GPU 1   ████████████│████████████████████████████│████████
GPU 2   ████████████│████████████████████████│░░░│████████
GPU 3   ████████████│████████████████████████████│████████
 ▲
 │
 slowest GPU
```

**Legend:**

- ████ = Forward
- ████ = Backward
- ░░░░ = GPU idle / waiting
- ████ = Communication
  The vertical marker shows the synchronization point.

#### Gradient Bucketing

Based on the research paper experiments (arXiv:2006.15704, Figure 2), when the number of parameters per `all-reduce` operation is small, the communication overhead is significant. In contrast, when the number of parameters per `all-reduce` operation is large, the total execution time is reduced. These experiments suggest that instead of running `all-reduce` as soon as the gradient tensors become available, we can wait a short time and accumulate the gradients into a bucket, and then run `all-reduce` on the bucket to reduce the communication overhead. However, we should not wait too long for the gradient tensors because models with large parameters need more time to compute the backward pass.

With an appropriate bucket size, we can overlap the communication with the backward pass, which is a significant performance optimization.

#### Overlapping Computation with Communication

The `all-reduce` operation can run concurrently with the backward pass. With the gradient bucketing technique, we can start the `all-reduce` operation on the first bucket of gradients while the backward pass is still computing the gradients for the remaining buckets. So, under these settings, the solution of waiting for the backward pass to finish before starting the `all-reduce` is no longer sufficient. Instead, we can start the `all-reduce` operation as soon as the first bucket of gradients is ready (PyTorch creates the autograd hooks to check the readiness of each gradient. If all gradients in the first bucket of all processes are ready, the `all-reduce` operation can be started), and continue to compute the remaining gradients while the `all-reduce` operation is in progress. This leads to two cautions:

1. The reducing order must be the same across all processes; otherwise, the results of `all-reduce` will be incorrect, or potentially the program will crash. The DDP module ensures that the reducing order is the same across all processes by using a deterministic order of gradient buckets.
    - Solution: PyTorch v1.5.0 addresses this problem by using the reverse order of model.parameters() as the bucketing order, assuming that layers are likely registered according to the same order as they are invoked in the forward pass. Hence, the reverse order (backward pass direction) should approximately represent the gradient computation order in the backward pass. Admittedly, this is not a perfect solution, but it is an approximation that we can rely on with minimum engineering overhead
2. It is possible that one training iteration only involves a sub-graph in the model, and the sub-graph can be different from iteration to iteration, meaning that some gradients might be skipped in some iterations (such as `Dropout`, `RNN`, `Transformer`, etc.). However, as gradient-to-bucket mapping is determined at construction time, those absent gradients would leave some buckets never seeing the final autograd hook and failing to mark the bucket as ready.
    - Solution: DDP traverses the autograd graph from the output tensors of the forward pass to find all participating parameters. The readiness of those participating tensors is a sufficient signal to conclude the completion of the backward pass. Therefore, DDP can avoid waiting for the rest of the parameter gradients by proactively marking them ready at the end of the forward pass

**TL;DR**: There are two problems about overlapping computation and communication: (1) the reducing order must be the same across all processes, and (2) some gradients might be skipped in some iterations. The DDP module addresses these problems by using a deterministic order of gradient buckets and proactively marking absent gradients as ready at the end of the forward pass.

**CODE:** Algorithm 1: Distributed Data Parallel, page 5, arXiv:2006.15704

#### Gradient Accumulation

One technique in DDP to speed up training is to reduce the gradient synchronization frequency. Instead of launching the `all-reduce` operation at every iteration, we can accumulate gradients for multiple iterations (`n` local training steps) and then launch the `all-reduce` operation.

**Use cases:** This is helpful in cases where the batch is too large and cannot fit into the GPU memory. The batch can be split into `n` smaller batches, and the gradients can be accumulated for each smaller batch. After `n` iterations, the `all-reduce` operation can be launched to synchronize the gradients across all devices. This technique can reduce the communication overhead and improve training throughput. Theoretically, this should produce the same results as if all data in the large batch is processed in one shot, as gradients will simply be accumulated to the same tensor.

However, this technique conflicts with the `gradient reduction` technique. Moreover, DDP cannot distinguish whether the application plans to immediately invoke `optimizer.step()` after backward or accumulate gradients through multiple iterations. Therefore, we need to introduce one additional interface (i.e., `no_sync`) for this use case.

### Communication Collectives

The connections between devices use point-to-point communication, and the communication collectives are built on top of these connections. There are three main communication collectives used in DDP:

1. _Gloo_: A collective communications library that supports CPU and GPU communication. It is optimized for low-latency and high-throughput communication.
2. _NCCL_: A collective communications library that supports GPU communication. It is optimized for high-bandwidth and low-latency communication on NVIDIA GPUs.
3. _MPI_: A message-passing interface that supports CPU and GPU communication. It is optimized for high-performance computing clusters.

## Extended Information

In the research paper (arXiv:2006.15704), the authors proposed two ways to update the model parameters:

1. **Synchronizing Gradients**: In this approach, the model computes gradients on each worker and then aggregates them from all workers (0 to N-1 devices/GPUs) - typically using an all-reduce operation (average gradients).
2. **Parameter Averaging**: In this approach, the model computes gradients on each worker and then updates the model parameters on each worker independently. After that, the model parameters are averaged across all workers (0 to N-1 devices/GPUs), typically using an all-reduce operation (average parameters).
    - This method has a problem: the results of distributed training are not mathematically equivalent to local training, because the local optimizer states or optimizer depends on the local past gradients (e.g., momentum, adaptive learning rates). Also, the structure of parameter averaging orchestrates computation (i.e., backward pass) and communication (i.e., computing average) into non-overlapping phases, using optimizer `step()` functions as a hard separation point. Regardless of how vigorously we optimize the computation or communication, one type of resource will stay idle at any given time instance, giving up a substantial performance optimization opportunity.

|                                    | Synchronizing Gradients                                                                     | Parameters Averaging                                  |
|------------------------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------|
| What gets combined?                | Gradients, before the optimizer step                                                        | Updated model parameters, after local optimizer steps |
| What does each worker update with? | The same combined gradient                                                                  | Its own local gradient                                |
| Do workers stay aligned?           | They apply the same gradient to the same starting parameters, so their updates stay aligned | Their local optimizer states and updates can diverge  |

The research paper (arXiv:2006.15704) prefers the first approach (Synchronizing Gradients) because it is mathematically equivalent to local training, while the second approach (Parameter Averaging) is not. The paper also provides a theoretical analysis of the convergence of both approaches and shows that Synchronizing Gradients has better convergence properties than Parameters Averaging.
