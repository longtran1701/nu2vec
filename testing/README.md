## Testing Overview

To show how well our method, multiple network node2vec, performs, we will
evaluate its performance in the task of function prediction versus
a few different methods.

The other methods evaluated are listed below:

- Simple majority vote
- Weighted majority vote
- K-nearest neighbors across various embeddings
    - Standard node2vec embedding
    - DSD embedding

### Function Prediction Module

This module performs function prediction on the (embedded) network, using
one of several algorithms. Module name is fpredict.

Inputs:

- Node labels file
    - Space seperated file with first item as node name and remaining
      as labels.
    - Example: examples/MIPSFirstLevel.anno3
- Network file
    - Either a R^d network embedding (same format as node2vec), the
      network itself as an edge list, or the STRING network.
- Function prediction algorithm
    - Majority vote (--mv)
    - Weighted majority vote (--wmv)
    - K-nearest neighbors (--knn K)
    - STRING majority vote (--string COLUMN)

Outputs:

Node label association file.

An example always helps to elucidate what is going on. This example
uses the k-nearest neighbors voting strategy with k = 2. The input is
an example embedding in R^2 and an example labelling. Every node is
labeled except for node E.

Input: 

```
python fpredict.py examples/faker2_emb.txt examples/faker2_labels.txt --knn 2
```

Ouput:

```
A Mouse Cat
B Mouse Cat
C Cat Dog
D Mouse Dog
E Bubbaloo
F Bubbaloo
```

### Evaluation Strategy

The perfomance measurement strategy is very simple. We will do
it using 2-fold cross validation. We will randomly remove half of the labels,
run function prediction on it, and evaluate the accuracy.

We want to test the performance of our algorithm on the STRING protein
interaction database.
