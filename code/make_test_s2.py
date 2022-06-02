import argparse
import os
import ipdb
import random
from tqdm import tqdm
import time

random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('clean input file')
    parser.add_argument('--inp_s1_predictions', type=str)
    parser.add_argument('--inp_sentences', type=str)
    # parser.add_argument('--data_type', type=str, required=True, help='model type')
    # parser.add_argument('--lang', type=str, required=True, help='language')
    parser.add_argument('--out_s2_input', type=str)
    args = parser.parse_args()

    from os.path import expanduser
    home = expanduser("~")

    # data_dir = home+"/moie_bucket/data"
    # args.fp1 = f"{data_dir}/{args.lang}/gen2oie_s1/{args.data_type}/test.predicted"
    # args.fp2 = f"{data_dir}/carb/data/{args.lang}_test.input"
    # args.out = f"{data_dir}/{args.lang}/gen2oie_s2/{args.data_type}/"
    with open(args.inp_s1_predictions, 'r') as f:
        relations = f.readlines()

    with open(args.inp_sentences, 'r') as f:
        sentences = f.readlines()

    assert len(relations) == len(sentences)

    def get_relations(x):
        res = []
        for r in x.split('<r>'):
            if r.strip() != "":
                res.append(r.strip())
        return res

    test_relations = []
    count_relations = []
    total_num_relations = 0
    for i in range(len(sentences)):
        all_relations = get_relations(relations[i])
        count_relations.append(str(len(all_relations)))
        total_num_relations += len(all_relations)
        for rel in all_relations:
            test_relations.append(rel.strip() + ' <r> ' + sentences[i].strip())

    with open(args.out_s2_input+'.s2_input', 'w') as f:
        f.write("\n".join(test_relations).strip())
    with open(args.out_s2_input+'.s2_count', 'w') as f:
        f.write("\n".join(count_relations).strip())

    print('Total num relations = ', total_num_relations)
