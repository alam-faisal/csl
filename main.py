from qaravan.tensorQ import op_action
from qaravan.core import ChiralHeisenberg
import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

def interp_ham(row_layout, sp, theta): 
    j = 1 
    jx = j * np.cos(theta)
    jy = jx
    jz = jx
    jc = j * np.sin(theta)
    return ChiralHeisenberg(row_layout, jx, jy, jz, jc, sp=sp)

def ham_action(grouped_terms, sv, local_dim=2): 
    result = np.zeros_like(sv, dtype=complex)
    for indices, mat in grouped_terms: 
        result += op_action(mat, indices, sv, local_dim=local_dim)
    return result

def ground(ham):
    grouped_terms = ham.grouped_terms()
    dim = 2**ham.num_sites

    def mv(v):
        return ham_action(grouped_terms, v)

    H_linop = LinearOperator(shape=(dim, dim), matvec=mv, dtype=np.complex128)
    vals, vecs = eigsh(H_linop, k=1, which='SA')  # 'SA' = smallest algebraic
    return vals[0], vecs[:, 0]