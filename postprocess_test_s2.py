import argparse
import os
import ipdb
import random
from tqdm import tqdm
import time

random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('clean input file')
    parser.add_argument('--fp1', type=str)
    parser.add_argument('--fp2', type=str)
    parser.add_argument('--data_type', type=str, required=True, help='data type')
    parser.add_argument('--lang', type=str, required=True, help='language')
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    from os.path import expanduser
    home = expanduser("~")
    data_dir = home+"/moie_bucket/data"

    args.fp1 = f"{data_dir}/{args.lang}/gen2oie_s2/{args.data_type}/test.count"
    args.fp2 = f"{data_dir}/{args.lang}/gen2oie_s2/{args.data_type}/test.pre_predicted"
    args.out = f"{data_dir}/{args.lang}/gen2oie_s2/{args.data_type}/test.predicted"
    with open(args.fp1, 'r') as f:
        count_relations = f.readlines()

    with open(args.fp2, 'r') as f:
        test_relations = f.readlines()
    print(len(count_relations))
    ind = 0
    output = []
    for i in range(len(count_relations)):
        count = int(count_relations[i].strip())
        sent_exts = []
        for ind1 in range(ind, ind+count):
            ext = test_relations[ind1].strip()
            sent_exts.append(ext)
        ind+=count
        if len(sent_exts) == 0:
            output.append('<e>')
        else:
            output.append(" ".join(sent_exts).strip())
    print(ind)
    with open(args.out, 'w') as f:
        f.write("\n".join(output).strip())
