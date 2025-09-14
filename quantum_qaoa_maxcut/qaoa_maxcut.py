# Quantum Approximate Optimization Algorithm (QAOA) for Max-Cut Problem
# Author: Sanjith Ram V
# Description:
#   This script demonstrates solving the Max-Cut problem using QAOA in Qiskit.
#   Max-Cut is a graph optimization problem where we partition vertices into two sets
#   such that the number of edges between the sets is maximized.
#   We compare QAOA results with a classical brute-force solver.

import networkx as nx
import matplotlib.pyplot as plt
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization.applications.ising import max_cut
import itertools

# -------------------------------
# Step 1: Define the graph
# -------------------------------
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])  # 4-node cycle + diagonal

# Draw the graph
nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray")
plt.title("Max-Cut Graph Example")
plt.show()

# -------------------------------
# Step 2: Convert to QUBO
# -------------------------------
qubo = max_cut.get_operator(G)

# -------------------------------
# Step 3: Define Quadratic Program
# -------------------------------
qp = QuadraticProgram()
for i in range(len(G.nodes)):
    qp.binary_var(name=f"x{i}")

# Objective: maximize cut edges (converted to minimize for Qiskit)
objective = {}
for edge in G.edges:
    objective[(f"x{edge[0]}", f"x{edge[1]}")] = 1
qp.minimize(linear=[], quadratic=objective)

print("Quadratic Program (QUBO Formulation):")
print(qp.export_as_lp_string())

# -------------------------------
# Step 4: Setup QAOA
# -------------------------------
backend = Aer.get_backend("qasm_simulator")
qaoa = QAOA(optimizer=COBYLA(maxiter=200), reps=1, quantum_instance=backend)

meo = MinimumEigenOptimizer(qaoa)

# Solve with QAOA
qaoa_result = meo.solve(qp)

print("\n=== QAOA Solution ===")
print("Optimal solution:", qaoa_result.x)
print("Objective value (cut size):", qaoa_result.fval)

# -------------------------------
# Step 5: Classical Brute Force Solver (for comparison)
# -------------------------------
def brute_force_maxcut(graph):
    n = len(graph.nodes)
    best_cut = None
    best_value = -1
    for bits in itertools.product([0, 1], repeat=n):
        cut_value = 0
        for u, v in graph.edges:
            if bits[u] != bits[v]:
                cut_value += 1
        if cut_value > best_value:
            best_value = cut_value
            best_cut = bits
    return best_cut, best_value

classical_solution, classical_value = brute_force_maxcut(G)

print("\n=== Classical Brute-Force Solution ===")
print("Best cut assignment:", classical_solution)
print("Maximum cut value:", classical_value)

# -------------------------------
# Step 6: Comparison
# -------------------------------
print("\n=== Comparison ===")
print(f"QAOA Cut Value: {qaoa_result.fval}")
print(f"Classical Max-Cut Value: {classical_value}")
