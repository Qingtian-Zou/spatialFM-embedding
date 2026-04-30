import numpy as np


def create_row_index_tensor(csr_matrix):
    indptr = csr_matrix.indptr
    row_indices = []
    for row in range(len(indptr) - 1):
        start = indptr[row]
        end = indptr[row + 1]
        row_indices.extend([row] * (end - start))
    return np.array(row_indices)
