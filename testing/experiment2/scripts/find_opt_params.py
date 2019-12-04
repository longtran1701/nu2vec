import argparse
import re
from tqdm import tqdm
from subprocess import Popen, PIPE
from os import listdir
from os.path import isfile, join

def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiment2.")
    parser.add_argument('--input', nargs='?', help="Input Folder of embeddings")

    return parser.parse_args()

def get_files(folder_name):
    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    return onlyfiles

def find_optimal_params(files, folder_name):
    acc_list = []
    for file in tqdm(files):
        name_list = file.split(".p.")[1]
        try:
            p, rest = tuple(name_list.split(".q."))
            q, rest = tuple(rest.split(".r."))
            r       = rest.split('.tmp.emb')[0]
            
            full_path = f'{folder_name}/{file}'
            args = ['python3', '../../fpredict.py', full_path, 
                '../inputs/mips-top-level.anno3', '--network-type', 'embedding', 
                '--algorithm', 'knn', '--args', '10', '--cross-validate', '2']
            print(' '.join(args))
            with open(full_path, 'r') as f:
                out, _ = Popen(args, stdout=PIPE).communicate()
                pattern = re.compile(r"Average Accuracy: (\d+\.\d+)")
                match = re.search(pattern, out.decode("ascii"))
                print("Here")
                if match is not None:
                    acc = match.group(1)
                    acc_list.append((p, q, r, float(acc)))
        except:
            print(f"Provided wrong file format: {file}")
            return None

    return max(acc_list, key=lambda x: x[3])



def main():
    args = parse_args()
    if args.input:
        files      = get_files(args.input)
        p,q,r,acc = find_optimal_params(files, args.input)
        print(f"Optimal Params Are \np: {p}\nq: {q}\nr: {r}\nWith Accuracy: {acc}")
    else:
        exit(0)



if __name__ == "__main__":
    main()