import os
import sys
import pickle
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.append(SRC_DIR)
from main import *

# ===== USER INPUT ===== #
gsfile = "gs12.pickle"
increments        = 8
layer_steps       = 7
num_instances     = 5

resume            = False
max_iter          = 5000
mag               = 1e-4
stop_ratio        = 1e-6
progress_interval = 20
checkpoint_interval = 50

# ===== Hamiltonian ===== #
row_layout = [{'num_cells': 1, 'shift': 0.0}, 
             {'num_cells': 2, 'shift': -0.5}, 
             {'num_cells': 3, 'shift': -1}]
sp = [0,1,9,11,16,17]
ham = interp_ham(row_layout, sp, 0.35 * np.pi)
ge, gs = ground(ham, quiet=True)   # let's do it manually this time

"""
with open(gsfile, "rb") as f:
    gs = pickle.load(f)
"""
    
# ===== RUN CONTEXT ===== #
jobname = os.path.basename(__file__).split(".")[0]
checkpoint_file  = f"{jobname}_checkpoint.pickle"    

context = RunContext(
    progress_interval=progress_interval,
    max_iter=max_iter,
    stop_ratio=stop_ratio,
    checkpoint_file=checkpoint_file, 
    checkpoint_interval=checkpoint_interval,
    resume=resume
)

# ===== OPTIMIZATION ===== #
filename = f"{jobname}.pickle"
best_cost, best_circs = train_with_layer_growth(
    gs, ham.lattice,
    increments=increments,
    layer_steps=layer_steps,
    filename=filename,
    num_instances=num_instances,
    mag=mag,
    context=context
)

print("Best costs:", [c[-1] for c in best_cost])