import torch


def generate_coo_data(size, sparse_dim, nnz, dtype):
    if dtype is None:
        dtype = 'float32'

    indices = torch.rand(sparse_dim, nnz)
    indices.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(indices))
    indices = indices.to(torch.long)
    values = torch.rand([nnz, ], dtype=dtype)
    return indices, values


def gen_sparse_input(size, density, dtype):
    sparse_dim = len(size)
    nnz = int(size[0] * size[1] * density)
    indices, values = generate_coo_data(size, sparse_dim, nnz, dtype)
    sparse = torch.sparse_coo_tensor(indices, values, size, dtype=dtype)
    dense = sparse.to_dense()
    return sparse, dense
