from qaravan.tensorQ import top_entropy
import pickle

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