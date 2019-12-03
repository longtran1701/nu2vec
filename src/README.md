## Getting Embeddings

We built a wrapper around node2vec to run our node2vec with a custom teleporatation parameter.

The format of the command is:
```
python3 man.py --input <input file> --keep <space-separated networks to keep> --p <p> --q <q> --r <r>
```

To test, we ran the command:
```
python3 man.py data.txt --keep cooccurence coexpression experimental --p 1 --q 1 --r 1
```