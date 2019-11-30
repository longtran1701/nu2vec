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
    - Either a R^d network embedding (same format as node2vec) or the
      network itself.
- Function prediction algorithm
    - Majority vote (--mv)
    - Weighted majority vote (--wmv)
    - K-nearest neighbors (--knn K)

Outputs:

Node label association file.

### Evaluation Strategy

The way we will evaluate is using 2-fold cross validation. 

    
