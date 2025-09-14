# Quantum Max-Cut with QAOA

This project demonstrates solving the **Max-Cut optimization problem** using the **Quantum Approximate Optimization Algorithm (QAOA)** with **Qiskit**.

## 📌 Problem
The **Max-Cut problem** partitions the nodes of a graph into two sets such that the number of edges between the sets is maximized.  
It has applications in scheduling, clustering, and network optimization.

## ⚡ Features
- Implements QAOA using Qiskit on a simple 4-node graph.
- Compares quantum solution with classical brute-force solution.
- Visualizes the graph with NetworkX + Matplotlib.

## 🚀 Requirements
Install dependencies:
```bash
pip install qiskit matplotlib networkx
```

## ▶️ Run the Script
```bash
python qaoa_maxcut.py
```

## 📊 Example Output
- Graph visualization (4 nodes with edges)
- QAOA solution (bitstring and objective value)
- Classical brute-force optimal cut
- Comparison between quantum and classical results

## 👤 Author
**Sanjith Ram V**  
B.Tech ECE, VIT Chennai  
Passionate about Quantum Computing, Cryptography, and HPC optimization.
