## Getting Embeddings

We built a wrapper around node2vec to run our node2vec with a custom teleporatation parameter.

The format of the command is:
```
python3 man.py --input <input file> --keep <space-separated networks to keep> --p <p> --q <q> --r <r>
```

To test, we ran the command:
```
python3 man.py --input data.txt --keep cooccurence coexpression experimental --p 1 --q 1 --r 1
```

To get a different r for each network, do the following:
```
python3 man.py --input data.txt --keep cooccurence coexpression experimental --p 1 --q 1 --rs 1 2 3
```

The program will interpret this as:
```
p = 1
q = 1
r_cooccurence  = 1
r_coexpression = 2
r_experimental = 3
```