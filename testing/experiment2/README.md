## Experiment 2 Overview

In this experiment, we want to train for the hyper-parameters *p*, *q*, and
*r* for function prediction using our multiple network embedding method.

We also want to get the baseline results using majority vote on the original
networks.


### Optimizing parameters

#### Dataset

The dataset for the networks consists of three YEAST protein networks
built from the STRING database. Namely, we will run our method on the
co-expression, co-occurrence, and experimental networks.

The dataset for the protein labels comes from the MIPS hierarchy root level.

#### Training

To train, we will perform a grid search over *p, q, r in {0.25, 0.5, 1, 2, 4}*.
The function that we will be optimizing is the average accuracy over 10-fold
cross validation on the task of function prediction.

The way we perform the function prediction is using majority vote on the
k-nearest neighbors under Euclidean distance. We set k = 10 and only look
at the neighbors that have labels.

#### Inputs

We have the following inputs stored under the inputs directory:

- (mips-top-level.anno3) MIPS top-level annotation file
- (string-yeast.networks) STRING networks

#### Outputs



### Baseline method results

We also ran simple majority vote and weighted majority vote with a 2-fold cross-validation using the following commands:

```
python3 fpredict.py experiment2/inputs/string-yeast.networks experiment2/inputs/mips-top-level.anno3 --network-type string --args <col-num> --algorithm <vote> --cross-validate 2
```

where

```python
<col-num> = 4 # --> cooccurence
          = 5 # --> coexpression
          = 6 # --> experimental


<vote> = mv  # simple majority vote
       = wmv # weighted majority vote
```

Results

|         | cooccurence - mv | cooccurence - wmv | coexpression - mv | coexpression - wmv | experimental - mv | experimental - wmv |
| ------- | ---------------- | ----------------- | ----------------- | ------------------ | ----------------- | ------------------ |
| 2-fold  | 0.74520678396517 | 0.745206783965173 | 0.740489379869999 | 0.6297536890114045 | 0.740489379869999 | 0.8059851122437773 |

This result is because we're using MIPS level 3 instead of level 1. However even
when we use MIPS level 1 to vote, the result is still absurdly good on weighted
majority vote (roughly 72% accuracy).