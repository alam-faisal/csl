from qaravan.tensorQ import top_entropy
import pickle

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