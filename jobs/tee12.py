import os
import sys
import pickle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.append(SRC_DIR)
from main import *

state_file = "12qubit_gs.pickle"
with open(state_file, "rb") as f:
    ge, gstate = pickle.load(f)

print("Ground energy:", ge)
A = [8,9]
B = [2,3]
C = [4,6]

tee = top_entropy(gstate, [A, B, C])
print("Topological entanglement entropy:", tee)

with open("12tee.pickle", "wb") as f:
    pickle.dump(tee, f)