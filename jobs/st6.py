import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.append(SRC_DIR)
from main import *

filename = "st6.pickle"
row_layout = [{'num_cells': 2, 'shift': -0.5}, 
             {'num_cells': 1, 'shift': 0.0}]
sp = [0,4,8]
ham = interp_ham(row_layout, sp, 0.35 * np.pi)
ge, gs = ground(ham, quiet=True)

best_cost, best_circs = train_with_layer_growth(gs, ham.lattice, increments=2, layer_steps=2, filename=filename, gates_per_layer=5, num_instances=1, max_iter=2000, mag=1e-2)
print("Best cost:", [c[-1] for c in best_cost])