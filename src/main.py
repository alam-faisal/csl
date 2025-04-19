from qaravan.tensorQ import op_action, rdm_from_sv, environment_state_prep
from qaravan.core import ChiralHeisenberg, vN_entropy, two_local_circ, RunContext
import numpy as np
from scipy.sparse.linalg import LinearOperator
from primme import eigsh
import sys
import pickle

#========== GS properties ============== #

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

def ground(ham, ncv=10, quiet=False):
    grouped_terms = ham.grouped_terms()
    dim = 2**ham.num_sites

    def mv(v):
        if v.ndim == 2 and v.shape[1] == 1:
            v = v[:, 0]
        return ham_action(grouped_terms, v)

    def convtest(eval_, evec, resNorm):
        print(f"Residual: {resNorm:.2e} for eigenvalue estimate: {eval_:.6f}")
        sys.stdout.flush()
        if np.abs(resNorm) < 1e-8: 
            return True
        return False
    
    ctest = convtest if not quiet else None
    H_linop = LinearOperator(shape=(dim, dim), matvec=mv, dtype=np.complex128)
    vals, vecs = eigsh(
        H_linop,
        k=1,
        which='SA',
        ncv=ncv,
        convtest=ctest,
    )
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

# ========== State Preparation ============== #

def edge_matching(nn_pairs, max_group_size=None):
    nn_pairs = list(nn_pairs)
    remaining = set(nn_pairs)
    groups = []

    while remaining:
        used_nodes = set()
        current_group = set()

        for a, b in sorted(remaining):
            if a not in used_nodes and b not in used_nodes:
                current_group.add((a, b))
                used_nodes.add(a)
                used_nodes.add(b)

                if max_group_size is not None and len(current_group) >= max_group_size:
                    break

        groups.append(list(current_group))
        remaining -= current_group

    return groups

def geometric_skeleton(lattice, gates_per_layer, num_layers): 
    groups = edge_matching(lattice.nn_pairs, max_group_size=gates_per_layer)
    skeleton = []
    for i in range(num_layers):
        skeleton.extend(groups[i%len(groups)])
    return skeleton

def sliding_skeleton(n, gates_per_layer, num_layers):
    skeleton = []
    start = 0

    while len(skeleton) < num_layers * gates_per_layer:
        layer = []
        used_qubits = set()

        for i in range(start, n - 1, 2):
            if len(layer) >= gates_per_layer:
                break
            layer.append((i, i + 1))
            used_qubits.update((i, i + 1))

        skeleton.extend(layer)

        if (n - 1) in used_qubits:
            start = 0
        else:
            start += 1

    return skeleton

def train(gs, lattice, num_layers, filename, gates_per_layer=5, context=None): 
    if context is None:    
        context = RunContext(progress_interval=200, max_iter=1000, stop_ratio=1e-6,
                             checkpoint_file=filename.replace(".pickle", "_checkpoint.pickle"))
    
    geom_skeleton = geometric_skeleton(lattice, gates_per_layer, num_layers=num_layers)
    current_circ = None
    if context.resume: 
        current_circ = context.opt_state['circ']
        context.log(f"Resuming with checkpointed circuit with {len(current_circ.gate_list)} gates")

    if current_circ is None: 
        circ, cost_list = environment_state_prep(gs, skeleton=geom_skeleton, context=context, quiet=True)
    else:
        circ, cost_list = environment_state_prep(gs, circ=current_circ, context=context, quiet=True)

    with open(filename, "wb") as f:
        pickle.dump((cost_list, circ), f)

    return cost_list, circ

def train_with_layer_growth(gs, lattice, increments, layer_steps, filename,
                            gates_per_layer=5, num_instances=5, mag=1e-3,
                            context=None):
    if context is None:    
        context = RunContext(progress_interval=200, max_iter=1000, stop_ratio=1e-6,
                             checkpoint_file=filename.replace(".pickle", "_checkpoint.pickle"))

    geom_skeleton = geometric_skeleton(lattice, gates_per_layer, num_layers=increments)
    current_circ = None
    if context.resume:
        current_circ = context.opt_state['circ']
        context.log(f"Resuming with checkpointed circuit with {len(current_circ.gate_list)} gates")

    best_cost, best_circs = [], []
    for i in range(layer_steps + 1):
        cost_against_instance, circ_against_instance = [], []

        for _ in range(num_instances):
            if current_circ is None:
                circ, cost_list = environment_state_prep(gs, skeleton=geom_skeleton, context=context)
            else:
                ansatz = current_circ + two_local_circ(geom_skeleton, mag=mag)
                circ, cost_list = environment_state_prep(gs, circ=ansatz, context=context)

            cost_against_instance.append(cost_list)
            circ_against_instance.append(circ)

        min_cost_index = np.argmin([cost_list[-1] for cost_list in cost_against_instance])
        best_cost.append(cost_against_instance[min_cost_index])
        current_circ = circ_against_instance[min_cost_index]
        best_circs.append(current_circ)

        with open(filename, "wb") as f:
            pickle.dump((best_cost, best_circs), f)

    return best_cost, best_circs