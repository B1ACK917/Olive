import pickle

import torch
import torch.utils.benchmark as benchmark_utils
from tqdm import tqdm

from utils.bench import gen_sparse_input


def main():
    tasks = [
        ("matmul", "torch.sparse.mm(x, y)", "torch.sparse.mm(sx, y)"),
        ("matmul", "torch.mm(x,y)", "torch.mm(dx,y)"),

        ("sum", "torch.sparse.sum(x,dim=0)", "torch.sparse.sum(sx,dim=0)"),
        ("sum", "torch.sum(x,dim=0)", "torch.sum(dx,dim=0)"),

        ("softmax", "torch.sparse.softmax(x,dim=0)", "torch.sparse.softmax(sx,dim=0)"),
        ("softmax", "torch.softmax(x,dim=0)", "torch.softmax(dx,dim=0)"),
    ]

    serialized_results, timers = [], []
    repeats = 2
    for label, sub_label, stmt in tasks:
        for density in [i / 100.0 for i in range(5, 100, 5)]:
            for size in [(8, 8), (32, 32), (64, 64), (128, 128)]:
                sx, dx = gen_sparse_input(size=size, density=density, dtype=torch.float32)
                y = torch.rand(size, dtype=torch.float32)
                timers.append(
                    benchmark_utils.Timer(
                        stmt=stmt,
                        globals={
                            "torch": torch,
                            "sx": sx,
                            "dx": dx,
                            "y": y,
                            "zero": torch.zeros(()),
                        },
                        label=label,
                        sub_label=sub_label,
                        env=str(f"density: {density}"),
                        description=f"size: {size}"
                    )
                )

    for timer in tqdm(timers * repeats, postfix="Benchmarking"):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))

    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])

    comparison.print()


if __name__ == "__main__":
    main()
