from qaravan.tensorQ import op_action, rdm_from_sv
from qaravan.core import ChiralHeisenberg, vN_entropy
import numpy as np
from scipy.sparse.linalg import LinearOperator
from primme import eigsh

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

def ground(ham, ncv=10):
    grouped_terms = ham.grouped_terms()
    dim = 2**ham.num_sites

    def mv(v):
        if v.ndim == 2 and v.shape[1] == 1:
            v = v[:, 0]
        return ham_action(grouped_terms, v)

    H_linop = LinearOperator(shape=(dim, dim), matvec=mv, dtype=np.complex128)
    vals, vecs = eigsh(H_linop, k=1, which='SA', ncv=ncv, maxiter=dim)  # 'SA' = smallest algebraic
    return vals[0], vecs[:, 0]

def top_entropy(gstate, regions): 
    rdm_A = rdm_from_sv(gstate, regions[0])
    rdm_B = rdm_from_sv(gstate, regions[1])
    rdm_C = rdm_from_sv(gstate, regions[2])
    rdm_AB = rdm_from_sv(gstate, regions[0]+regions[1])
    rdm_BC = rdm_from_sv(gstate, regions[1]+regions[2])
    rdm_AC = rdm_from_sv(gstate, regions[0]+regions[2])
    rdm_ABC = rdm_from_sv(gstate, regions[0]+regions[1]+regions[2])

    sA = vN_entropy(rdm_A)
    sB = vN_entropy(rdm_B)
    sC = vN_entropy(rdm_C)
    sAB = vN_entropy(rdm_AB)
    sBC = vN_entropy(rdm_BC)
    sAC = vN_entropy(rdm_AC)
    sABC = vN_entropy(rdm_ABC)
    return np.real(sA + sB + sC - sAB - sBC - sAC + sABC)