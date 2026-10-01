# How to implement DDP with PyTorch?

Distributed Data Parallel (DDP) is a module wrapper that enables easy and efficient multi-GPU training in PyTorch. It is designed to work with multiple processes, each of which can run on a different GPU.

## Assumptions

You need to set up your environment with the following assumptions:

- You have a working installation of PyTorch with GPU support.
- You have multiple GPUs available for training.

## Set up environment

To use DDP, you need to set up your environment properly. Configure environment variables for distributed training.

Explanation of the environment variables:

- `backend`: The backend takes responsibility for communicating between processes. It is used for distributed training. Options include "nccl" (recommended for NVIDIA GPUs), "gloo" (for CPU and GPU), and "mpi" (for MPI-based communication).
- `master_addr`: The address of the master node (Rank 0; the master GPU runs the source code). It is used to establish communication between processes. For single-node training, you can set it to "localhost".
  - You can set it to the IP address of the master node in a multi-node setup such as `MASTER_ADDR=129.0.0.1`.
  - You can set it to the hostname of the master node in a multi-node setup such as `MASTER_ADDR=master-node`.
- `master_port`: The port number on which the master node will listen for incoming connections. It should be a free port that is not used by other processes. By convention, the PyTorch DDP uses port `29500` by default, but you can choose any available port.
- `world_size`: The total number of processes (GPUs) that will participate in the distributed training. It should be equal to the number of GPUs you want to use.
- `rank`: The rank of the current process. It should be a number between 0 and `world_size-1`.

```yaml
model:
    name: "YourModelName"
    type: "YourModelType"
    ddp:
        backend: "nccl" # Options: "nccl", "gloo", "mpi"
        master_addr: "localhost" 
        master_port: 12345 
        world_size: 4 # Total number of processes (GPUs) to use
        rank: 0 # Rank of the current process (0 to world_size-1)
```

## Initialize the process group

To initialize the process group, you need to call `torch.distributed.init_process_group()` at the beginning of your training script. This function sets up the communication between processes and initializes the distributed environment.

```python
import torch
import torch.distributed as dist

def init_process_group(backend, master_addr, master_port, world_size, rank):
    dist.init_process_group(
        backend=backend,
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )
```

## Set up sampler and data loader

When using DDP, you need to ensure that each process gets a different subset of the dataset. This is typically done by using `torch.utils.data.distributed.DistributedSampler`, which ensures that each process only sees a portion of the dataset.

```python
from torch.utils.data import DataLoader, DistributedSampler

def get_data_loader(dataset, batch_size, rank, world_size):
 sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
 data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return data_loader
```

## Initialize the multiple processes

To initialize multiple processes for DDP, you can use the `torch.multiprocessing.spawn()` function. This function will spawn multiple processes, each of which will run the training script with its own rank.

```python
import torch.multiprocessing as mp

def train(rank, world_size):
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```
