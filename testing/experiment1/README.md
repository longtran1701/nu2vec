Experiment 1
============

Perform 2 fold cross validation on regular node2vec embeddings
and node2vec multi plex embeddings.

The embedding files have the "4932." part removed from each protein name.

File ReformattedMIPSLabels.anno3 is the MIPSLabels.anno3 file where
all entries without labels are removed.

That is, a row like:

```
PROTEINX
```

Is deleted.

The command to perform the cross validation used in the experiment is:

```
python fpredict.py experiment1\ce.exp.co.4932.emb experiment1\ReformattedMIPSLabels.anno3 --cross-validate 2 --network-type embedding --algorithm knn --args 10
```

This runs knn majority vote on the 10 nearest neighbords with labels and performs 2 cross fold
validation.

All files used are in this directory.