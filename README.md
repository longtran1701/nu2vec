# nu2vec

This repository provides a reference implementation of **nu2vec** (Network Unification to Vectors) which is experiment to enhance [node2vec](https://github.com/aditya-grover/node2vec) on multiplexed networks.

### Problem

Given fixed vertex set $V$ and networks $G_1 = (V, E_1)$, ... , $G_D = (V, E_D)$

We want to find an embedding $T: V \rightarrow R^k$ that preserves properties across all the networks.

### Proposal

Modify the formulation of transition biases of node2vec to be:
$$
\alpha_{pqr}(t, v, x) = 
\begin{cases} 
    \frac{1}{p}         & \text{if } d_{tx} = 0 \\
    1                   & \text{if } d_{tx} = 1 \\
    \frac{1}{q}         & \text{if } d_{tx} = 2 \\
    \frac{1}{r (n - 1)} & \text{if } x \in neighbor(alias(v))
\end{cases}
$$
where parameter $r$ controls the network teleportation bias when performing random walks.


### How to run