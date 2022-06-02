import argparse
import os
import ipdb
from tqdm import tqdm
import stanza
import string

PUNCT = string.punctuation + '，' + '《' + '》'+'）' + '('
Count = 0
global helper_dict

def clean_extractions(ext):
    global Count
    exts = ext.split('<e>')[:-1]
    cleaned_exts = []
    for ext in exts:
        if ('<r>' in ext and '</r>' in ext) or ('<rel>' in ext and '</rel>' in ext):
            cleaned_exts.append(ext.strip())
    if len(cleaned_exts) == 0:
        Count+=1
        return '<a1> empty </a1> <e>'
    else:
        return " <e> ".join(cleaned_exts).strip() + ' <e> '

def generate_label(labels):
    global Count, helper_dict
    res = ''
    for label in labels:
        ans = ''
        sentence, tag = label.split('|||')
        sent = sentence.strip().split()
        tag = tag.strip().split()
        assert len(sent) == len(tag), ipdb.set_trace()
        ind = 0
        while ind < len(sent):
            ans_t = ''
            if tag[ind] != 'NONE':
                prev_label = tag[ind]
                ans_t_tag = ''
                while ind < len(sent) and tag[ind] == prev_label:
                    ans_t_tag += (sent[ind] + ' ')
                    ind+=1
                ans_t_tag = ans_t_tag.strip().strip(PUNCT).strip()
                # ans_t_tag = ans_t_tag.strip()
                if ans_t_tag != "":
                    bound_tag = helper_dict[prev_label]
                    ans_t += (bound_tag[0] + ' ')
                    ans_t += (ans_t_tag + ' ' + bound_tag[1])
                    ans += (ans_t + ' ')
            else:
                ind+=1
        if ans == "":
            continue
        ans += '<e> '
        res+=ans
    if res.strip() == "":
        Count+=1
        return sentence, '<a1> empty </a1> <e>'
    else:
        res = clean_extractions(res)
    return sentence, res.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('clean input file')
    parser.add_argument('--fp1', type=str, required=True, help='input file')
    parser.add_argument('--fp2', type=str, required=True, help='input file')
    parser.add_argument('--out', type=str, required=True, help='output file')
    parser.add_argument('--conf', type=str, help='confidence file')
    parser.add_argument('--type', type=str, help='allennlp/None')
    args = parser.parse_args()

    global helper_dict
    if args.type == 'allennlp':
        helper_dict = {'REL': ['<rel>', '</rel>'], 'ARG1': ['<arg1>', '</arg1>'], 'ARG2':['<arg2>', '</arg2>']}
        assert args.conf != None
    else:
        helper_dict = {'REL': ['<r>', '</r>'], 'ARG1': ['<a1>', '</a1>'], 'ARG2':['<a2>', '</a2>'],\
    'LOC': ['<l>', '</l>'], 'TIME': ['<t>', '</t>']}

    with open(args.fp1, 'r') as f1,\
        open(args.fp2, 'r') as f2,\
        open(args.out, 'w') as f3:
            data1 = f1.readlines()
            data2 = f2.readlines()
            if args.conf:
                confidences = open(args.conf,'r').readlines()
                assert len(data1) == len(confidences) == sum([int(l) for l in data2])
            index = 0
            for ind in range(len(data2)):
                count = int(data2[ind].split('\t')[0].strip())
                if count == 0:
                    continue
                sentence, label = generate_label(data1[index:index+count])
                if args.type == 'allennlp':
                    for ext in label.split('<e>')[:-1]:
                        if args.conf:
                            f3.write(sentence+'\t'+ext.strip()+'\t'+confidences[index].strip()+'\n')
                        else:
                            f3.write(sentence+'\t'+ext.strip()+'\t1\n')
                else:
                    f3.write(label + '\n')
                index += count

    print('empty extractions are', Count)

