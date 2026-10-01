import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def setup(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(0)


def cleanup() -> None:
    dist.destroy_process_group()


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        return self.fc(x)


class SyntheticImages(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic 28x28 images and labels shared by every rank."""

    def __init__(self, n: int = 2000, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 1, 28, 28, generator=generator)
        self.y = torch.randint(0, 10, (n,), generator=generator)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def train(rank: int, world_size: int, epochs: int = 2) -> None:
    distributed = world_size > 1
    if distributed:
        setup(rank, world_size)
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dataset = SyntheticImages()
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=0,
        )
        loader = DataLoader(dataset, batch_size=64, sampler=sampler)

        torch.manual_seed(0)
        model = TinyCNN().to(device)
        if distributed:
            model = DDP(model, device_ids=[0])
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            running_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(
                f"[rank {rank}] epoch {epoch} "
                f"avg loss {running_loss / len(loader):.4f}"
            )

        checksum = sum(
            parameter.detach().sum().item() for parameter in model.parameters()
        )
        print(f"[rank {rank}] parameter checksum after training: {checksum:.6f}")
    finally:
        if distributed:
            cleanup()


def run_ddp_demo(world_size: int = 1, epochs: int = 2) -> None:
    mp.spawn(train, args=(world_size, epochs), nprocs=world_size, join=True)


def main() -> None:
    run_ddp_demo(world_size=2, epochs=10)


if __name__ == "__main__":
    main()
