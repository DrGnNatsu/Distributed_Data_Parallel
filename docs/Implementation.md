# How to implement DDP with PyTorch?

Distributed Data Parallel (DDP) is a module wrapper that enables easy and efficient multi-GPU training in PyTorch. It is designed to work with multiple processes, each of which can run on a different GPU.

Flow of DDP training:

The conceptual flow is:

```text
Initialize  environment --> Get rank / local_rank / world_size --> Select GPU --> Initialize process group --> Create Dataset --> Create DistributedSampler --> Create DataLoader --> Create Model --> Move Model to local GPU --> Wrap Model with DDP --> Training --> Checkpoint/log on rank 0 --> destroy_process_group()
```

## 1. Requirements

- You have a working installation of PyTorch with GPU support.
- You have multiple GPUs available for training.

## 2. DDP Concepts

### Process–GPU Relationship

DDP normally uses:

```text
1 process ↔ 1 GPU
```

Therefore, in a typical single-node multi-GPU configuration:

```text
world_size = number of processes = number of GPUs
```

Strictly speaking, `world_size` represents the **total number of distributed processes**, not GPUs themselves.

### Key Concepts

- **Process**: An independent Python process running the training script
- **GPU**: Physical GPU device assigned to a process
- **Rank**: Global identifier of a process (0 to world_size-1)
- **Local Rank**: Local identifier of a process on its machine (0 to nproc_per_node-1)
- **World Size**: Total number of distributed processes in the process group
- **Process Group**: Communication group for distributed operations
- **Distributed Sampler**: Ensures each process sees a different subset of the dataset

## 3. Single-Node Multi-GPU Setup

Recommended Launch Command

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

For a machine with four GPUs, this starts four training processes.

If you do not use `torchrun`, you need to manually use the `mp.spawn` API to launch multiple processes, which is more complex and error-prone.

**NOTE**: Do not use two methods simultaneously. Use either `torchrun` or `mp.spawn`, but not both.

Explanation of the `torchrun` command:

- `--standalone`: Indicates that this is a single-node run. For multi-node runs, you would specify the number of nodes and the rank of each node. If we do not have this flag, we need to configure the rendezvous endpoint (`master_addr`, `mater_port`, and `node_rank`) for the distributed processes to communicate.
- `--nproc_per_node`: The number of processes to launch on each node.

## 4. Distributed Environment

When using `torchrun`, the following environment variables are automatically set:

- `RANK`: Global rank of the current process (`0` to `world_size-1`)
- `LOCAL_RANK`: Local identifier on the current machine, used to select the GPU (`0` to `nproc_per_node-1`)
- `WORLD_SIZE`: Total number of distributed processes participating in the process group
- `MASTER_ADDR`: Network address of the rendezvous endpoint, normally associated with the machine running rank 0. (default: `127.0.0.1`)
- `MASTER_PORT`: TCP port for the rendezvous endpoint. Must be available and the same for all processes. (default: `29500`)

The `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` should not hard configured in the configuration file. They are automatically set by the launcher and should be obtained from the environment variables. In contrast, `MASTER_ADDR` and `MASTER_PORT` can be set in the configuration file or environment variables, but they must be consistent across all processes.

### Obtaining Distributed Information

```python
import os

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
```

This lets the launcher assign the correct distributed information to each process.

## 5. Initialize the Process Group

```python
import torch
import torch.distributed as dist

def init_process_group(backend="nccl"):
    dist.init_process_group(backend=backend)
```

For NVIDIA GPUs, `"nccl"` is the recommended backend.

## 6. Assign GPUs

Each process needs to know which GPU on its local machine it should use:

```python
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
```

Typical mapping:

```text
rank 0 → cuda:0
rank 1 → cuda:1
rank 2 → cuda:2
rank 3 → cuda:3
```

## 7. Dataset and DistributedSampler

When using DDP, you need to ensure that each process gets a different subset of the dataset:

```python
from torch.utils.data import DataLoader, DistributedSampler

def get_data_loader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return data_loader
```

### Important: Set Epoch for Shuffling

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)
    for batch in dataloader:
        ...
```

This ensures that the sampler's shuffling changes correctly between epochs.

## 8. DataLoader

The `batch_size` passed to each process's `DataLoader` is the **per-GPU/per-process batch size**.

For example:

```python
batch_size = 8
```

with four GPUs means approximately:

```text
Global batch size = batch size per GPU × number of GPUs = 8 × 4 = 32
```

This distinction is important because changing the number of GPUs can change the effective global batch size and therefore potentially affect training dynamics and learning-rate choices.

## 9. Model and DistributedDataParallel Wrapper

The model should be assigned to its local GPU and wrapped with DDP:

```python
model = YourModel().to(local_rank)

model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank]
)
```

## 10. Checkpointing

In DDP, all processes execute the training code. Therefore, saving a checkpoint from every process can result in duplicated writes.

A common pattern is:

```python
if rank == 0:
    torch.save(model.state_dict(), "checkpoint.pth")
```

Only rank 0 should normally perform the main checkpoint/logging operation unless there is a specific reason for distributed checkpointing.

## 12. Logging

Because every DDP process executes the training loop, naïve logging can produce duplicated output:

```text
Rank 0: Epoch 1 ...
Rank 1: Epoch 1 ...
Rank 2: Epoch 1 ...
Rank 3: Epoch 1 ...
```

For ordinary training logs, it is often preferable to log only from rank 0:

```python
if rank == 0:
    print(f"Epoch {epoch}: Loss = {loss.item()}")
```

Metrics that require information from all processes may additionally require distributed reduction/aggregation.

## 13. Cleanup

After training, clean up the distributed process group:

```python
dist.destroy_process_group()
```

## 14. Complete Example

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def main():
    # Get distributed info from environment
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize process group
    dist.init_process_group(backend="nccl")

    # Dataset and sampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Checkpoint (rank 0 only)
    if rank == 0:
        torch.save(model.module.state_dict(), "mnist_ddp.pth")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Launch with:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

## 15. Common Problems

| Problem | Solution |
| --------- | ---------- |
| Address already in use | Change `MASTER_PORT` or ensure no other DDP runs are active |
| NCCL errors | Check GPU compatibility, driver versions, and network connectivity |
| Incorrect GPU assignment | Ensure `torch.cuda.set_device(local_rank)` is called before model creation |
| Dataset duplication | Use `DistributedSampler` and call `sampler.set_epoch(epoch)` |
| Incorrect batch size | Remember `batch_size` is per-GPU; global batch = batch_size × world_size |
| Processes hanging | Ensure all processes reach collective operations; check for rank-specific code paths |

## Summary of Key Points

| Area | Recommendation |
| --- | --- |
| Process launching | Prefer `torchrun --standalone --nproc_per_node=N` |
| GPU assignment | Use `local_rank` from environment |
| Rank/world size | Obtain from `RANK`, `LOCAL_RANK`, `WORLD_SIZE` env vars |
| DDP model | Wrap with `DistributedDataParallel(model, device_ids=[local_rank])` |
| Sampler | Use `DistributedSampler` and call `set_epoch()` each epoch |
| Batch size | Per-GPU batch size; global = per-GPU × world_size |
| Master node | Rendezvous endpoint (rank 0 machine), not "master GPU" |
| Master port | Rendezvous port, not gradient communication port |
| Checkpointing | Save from rank 0 only |
| Logging | Log from rank 0 only |
| Cleanup | Call `dist.destroy_process_group()` |

## Multinode Setup

We need to run the training script on multiple nodes. Each node can have multiple GPUs. The `torchrun` command can be used to launch the training script on each node. We need to run the following command on each node, changing the `--node_rank` argument to the appropriate value for each node (0, 1, 2, ...). The `--master_addr` and `--master_port` arguments should be set to the address and port of the master node (rank 0).

```bash
torchrun \
    --nnodes=3 \
    --nproc_per_node=4 \
    --node_rank=0 \ # The node rank is the node's index (0, 1, 2, ...) in the cluster
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    train.py
```
