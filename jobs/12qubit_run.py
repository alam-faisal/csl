import os
import sys
import pickle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.append(SRC_DIR)
from main import *

row_layout = [{'num_cells': 1, 'shift': 0.0}, 
             {'num_cells': 2, 'shift': -0.5}, 
             {'num_cells': 3, 'shift': -1}]
sp = [0,1,9,11,16,17]
ham = interp_ham(row_layout, sp, 0.35 * np.pi)

ge, gstate = ground(ham)
with open("12qubit_gs.pickle", "wb") as f:
    pickle.dump((ge, gstate), f)
print("Ground energy:", ge)