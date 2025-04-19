import os
import sys
import pickle
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.append(SRC_DIR)
from main import *

# ===== USER INPUT ===== #
gsfile = "gs24.pickle"
num_layers = 48

resume            = False
max_iter          = 5000
stop_ratio        = 1e-6
progress_interval = 10
checkpoint_interval = 50

# ===== SYSTEM SETUP ===== #
row_layout = [{'num_cells': 3, 'shift': 0.0}, 
              {'num_cells': 4, 'shift': -0.5}, 
              {'num_cells': 3, 'shift': 0.0}, 
              {'num_cells': 2, 'shift': 0.5}]
sp = [0,1,6,7,9,11,30,32,34,35,19,20]
ham = interp_ham(row_layout, sp, 0.35 * np.pi)

with open(gsfile, "rb") as f:
    gs = pickle.load(f)

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
best_cost, best_circ = train(
    gs, ham.lattice,
    num_layers=num_layers,
    filename=filename,
    context=context
)