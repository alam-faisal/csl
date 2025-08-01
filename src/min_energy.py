from main import * 
from qaravan.core import two_local_circ, set_style, RunContext 
from qaravan.tensorQ import StatevectorSim
import torch 
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

def circuit_energy(circ, grouped_terms): 
    sim = StatevectorSim(circ, backend="torch")
    sim.run(progress_bar=False)
    ansatz_sv = sim.get_statevector()
    return (ansatz_sv.conj() @ ham_action(grouped_terms, ansatz_sv)).real 

def circuit_energy_from_params(params, skeleton, grouped_terms): 
    circ = two_local_circ(skeleton, params)
    return circuit_energy(circ, grouped_terms)

def rel_energy_err(params, skeleton, grouped_terms, true_ge): 
    return abs(circuit_energy_from_params(params, skeleton, grouped_terms) - true_ge) / abs(true_ge)

def energy_min_run(num_layers, lattice, grouped_terms, true_ge, lr, max_iter=20000, gamma=0.5, patience=200): 
    skeleton = geometric_skeleton(lattice, 5, num_layers)
    params = torch.nn.Parameter(torch.randn(15 * len(skeleton), dtype=torch.float64))
    
    optimizer = torch.optim.AdamW([params], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',             
        factor=gamma,           
        patience=patience,      
        threshold=1e-4,         
        threshold_mode='rel'
    )

    context = RunContext(
        progress_interval=500,
        max_iter=max_iter,
        checkpoint_file="checkpoint.pickle",
        checkpoint_interval=500,
        resume=False,
        convergence_check=True,
        stop_ratio=1e-8
    )   
    
    cost_list = []
    run_state = {
        "cost_list": cost_list, 
        "step": context.step
    }

    while True: 
        cost = rel_energy_err(params, skeleton, grouped_terms, true_ge)
        cost_val = cost.item()
        cost_list.append(cost_val)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(cost_val)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"[LR Drop] Step {run_state['step']}: cost = {cost:.2e}, LR dropped from {old_lr:.2e} to {new_lr:.2e}")

        run_state["step"] += 1
        run_state["cost_list"] = cost_list

        if context.step_update(run_state): 
            break

    return cost_list, params

def parallel_run(seed_and_args):
    seed, num_layers, lattice, grouped_terms, true_ge, lr, gamma = seed_and_args
    torch.manual_seed(seed)
    cost_list, params = energy_min_run(num_layers, lattice, grouped_terms, true_ge, lr=lr, gamma=gamma)
    return {
        "seed": seed,
        "final_cost": cost_list[-1],
        "cost_list": cost_list,
        "params": params.detach().numpy()
    }

def run_multiple_parallel(num_runs, num_layers, lattice, grouped_terms, true_ge, lr, gamma=0.1):
    seeds = list(range(num_runs))
    args = [(seed, num_layers, lattice, grouped_terms, true_ge, lr, gamma) for seed in seeds]

    with mp.get_context("spawn").Pool(processes=min(num_runs, mp.cpu_count())) as pool:
        results = pool.map(parallel_run, args)

    return results

# Define the problem
row_layout = [{'num_cells': 1, 'shift': 0.0}, 
             {'num_cells': 2, 'shift': -0.5}, 
             {'num_cells': 3, 'shift': -1}]
sp = [0,1,9,11,16,17]
ham = interp_ham(row_layout, sp, 0.35 * np.pi)
grouped_terms = [(indices, torch.tensor(mat, dtype=torch.complex128)) for indices, mat in ham.grouped_terms()]
true_ge, true_gs = ground(ham, quiet=True)
print(f"True ground state energy: {true_ge}")


if __name__ == "__main__":
    results = run_multiple_parallel(
        num_runs=5,
        num_layers=4,
        lattice=ham.lattice,
        grouped_terms=grouped_terms,
        true_ge=true_ge,
        lr=1e-2, 
        gamma=0.1,
    )

    for r in results:
        print(f"Seed {r['seed']} → Final cost: {r['final_cost']:.6f}")
        with open(f"results_{r['seed']}.pickle", "wb") as f:
            pickle.dump((r["cost_list"], r["params"]), f)