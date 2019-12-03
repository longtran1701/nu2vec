## Usage of find_opt_params.py

``` shell
python3 find_opt_params.py --input [FOLDER_PATH]
```

Given a folder path containing embeddings in the form 

`*.p._.q._.r._.emb`

Will attempt to find highest accuracy yielding parameters p, q and r for the following instance of `fpredict.py`

```shell
python3 ../../fpredict.py [FILE] ../inputs/mips-top-level.anno3 --network-type embedding --algorithm knn --args 10 --cross-validate 2
```