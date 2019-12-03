## Experiment 2 Overview

In this experiment, we want to train for the hyper-parameters *p*, *q*, and
*r* for function prediction using our multiple network embedding method.

### Dataset

The dataset for the networks consists of three YEAST protein networks
built from the STRING database. Namely, we will run our method on the
co-expression, co-occurrence, and experimental networks.

The dataset for the protein labels comes from the MIPS hierarchy root level.

### Training

To train, we will perform a grid search over *p, q, r in {0.25, 0.5, 1, 2, 4}*.
The function that we will be optimizing is the average accuracy over 10-fold
cross validation on the task of function prediction.

The way we perform the function prediction is using majority vote on the
k-nearest neighbors under Euclidean distance. We set k = 10 and only look
at the neighbors that have labels.

### Inputs

We have the following inputs stored under the inputs directory:

- (mips-top-level.anno3) MIPS top-level annotation file
- (string-yeast.networks) STRING networks

### Outputs




