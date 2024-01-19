import pickle

import torch
import torch.utils.benchmark as benchmark_utils
from tqdm import tqdm
from model.resnet import ResNet50
from model.unet import UNet


def generate_sparse_matrix_2d(size, density, dtype):
    nnz = int(size[0] * size[1] * density)
    indices = torch.randint(0, size[0], (nnz,))
    indices = torch.cat([indices.unsqueeze(0), torch.randint(0, size[1], (nnz,)).unsqueeze(0)], dim=0)
    values = torch.randn(nnz)
    sparse_matrix = torch.sparse_coo_tensor(indices, values, size, dtype=dtype)
    return sparse_matrix


def generate_sparse_tensor_3d(size, density, dtype):
    nnz = int(size[0] * size[1] * size[2] * density)
    indices = torch.randint(0, size[0], (nnz,))
    indices = torch.cat([indices.unsqueeze(0), torch.randint(0, size[1], (nnz,)).unsqueeze(0),
                         torch.randint(0, size[2], (nnz,)).unsqueeze(0)], dim=0)
    values = torch.randn(nnz)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size, dtype=dtype)
    return sparse_tensor, indices, values


def bench(timers, repeats):
    serialized_results = []
    for timer in tqdm(timers * repeats, postfix="Benchmarking"):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))

    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])

    comparison.print()


def bench_base():
    tasks = [
        ("matmul", "torch.sparse.mm(x, y)", "torch.sparse.mm(sx, y)"),
        ("matmul", "torch.mm(x,y)", "torch.mm(dx,y)"),
        #
        # ("sum", "torch.sparse.sum(x,dim=0)", "torch.sparse.sum(sx,dim=0)"),
        # ("sum", "torch.sum(x,dim=0)", "torch.sum(dx,dim=0)"),
        #
        # ("softmax", "torch.sparse.softmax(x,dim=0)", "torch.sparse.softmax(sx,dim=0)"),
        # ("softmax", "torch.softmax(x,dim=0)", "torch.softmax(dx,dim=0)"),
    ]

    timers = []
    repeats = 2
    for label, sub_label, stmt in tasks:
        for density in [i / 100.0 for i in range(5, 100, 5)]:
            for size in [(8, 8), (32, 32), (64, 64), (128, 128)]:
                sx = generate_sparse_matrix_2d(size=size, density=density, dtype=torch.float32)
                dx = sx.to_dense()
                y = torch.rand(size, dtype=torch.float32)
                timers.append(
                    benchmark_utils.Timer(
                        stmt=stmt,
                        globals={
                            "torch": torch,
                            "sx": sx,
                            "dx": dx,
                            "y": y
                        },
                        label=label,
                        sub_label=sub_label,
                        env=str(f"density: {density}"),
                        description=f"size: {size}"
                    )
                )
    bench(timers, repeats)


def bench_resnet50():
    tasks = [
        ("ResNet 50", "Dense Implementation", "model(x)"),
    ]

    timers = []
    repeats = 2
    model = ResNet50(2)
    for label, sub_label, stmt in tasks:
        for density in [i / 100.0 for i in range(5, 100, 5)]:
            for size in [(3, 64, 64), (3, 128, 128), (3, 256, 256)]:
                sx, *_ = generate_sparse_tensor_3d(size=size, density=density, dtype=torch.float32)
                dx = sx.to_dense().unsqueeze(dim=0)
                timers.append(
                    benchmark_utils.Timer(
                        stmt=stmt,
                        globals={
                            "torch": torch,
                            "model": model,
                            "x": dx,
                        },
                        label=label,
                        sub_label=sub_label,
                        env=str(f"density: {density}"),
                        description=f"size: {size}"
                    )
                )
    bench(timers, repeats)


def bench_unet():
    tasks = [
        ("UNet", "Dense Implementation", "unet(x)"),
        # ("UNet", "Sparse Implementation", "sparse_unet(x)"),
    ]

    timers = []
    repeats = 2
    unet = UNet(3, 2)
    for label, sub_label, stmt in tasks:
        for density in [i / 100.0 for i in range(5, 100, 5)]:
            for size in [(3, 64, 64), (3, 128, 128), (3, 256, 256)]:
                sx, indices, values = generate_sparse_tensor_3d(size=size, density=density, dtype=torch.float32)
                dx = sx.to_dense().unsqueeze(dim=0)
                coo = [indices, values]
                timers.append(
                    benchmark_utils.Timer(
                        stmt=stmt,
                        globals={
                            "torch": torch,
                            "unet": unet,
                            "x": dx,
                        },
                        label=label,
                        sub_label=sub_label,
                        env=str(f"density: {density}"),
                        description=f"size: {size}"
                    )
                )
    bench(timers, repeats)
