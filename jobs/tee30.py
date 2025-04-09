import os
import sys
import pickle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
sys.path.append(SRC_DIR)
from main import *

state_file = "30gs.pickle"
with open(state_file, "rb") as f:
    ge, gstate = pickle.load(f)

print("Ground energy:", ge)
A = [18, 20, 21, 22]
B = [12, 13, 14, 23]
C = [4,9,10,11]

tee = top_entropy(gstate, [A, B, C])
print("Topological entanglement entropy:", tee)

with open("30tee.pickle", "wb") as f:
    pickle.dump(tee, f)