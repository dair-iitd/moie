import argparse
import os
import ipdb
from tqdm import tqdm
import string

punct = string.punctuation
def allennlp(sentence, extractions_str, lang):
    def helper(x, y1, y2):
        res = ''
        prev_x = x.copy()
        while y1 in x and y2 in x:
            start_index = x.index(y1)
            end_index = x.index(y2)
            x[start_index] = ''
            x[end_index] = ''
            for ind in range(start_index+1, end_index):
                if x[ind] in ['<r>', '</r>', '<a2>', '</a2>', '<l>', '</l>', '<t>', '</t>','<a1>', '</a1>']:
                    break
                if x[ind].strip() != "":
                    res += (x[ind].strip() + ' ')
        return res.strip()
    # assert  len(extractions_str.split()) == len(scores_str.split()), ipdb.set_trace()
    res = []
    extractions_list = extractions_str.split('<e>')
    for extraction in extractions_list:
        if extraction == '':
            continue
        extraction_split = extraction.strip().split()
        arg1 = helper(extraction_split, '<a1>', '</a1>')
        rel = helper(extraction_split, '<r>', '</r>')
        arg2 = helper(extraction_split, '<a2>', '</a2>')
        loc = helper(extraction_split, '<l>', '</l>')
        time = helper(extraction_split, '<t>', '</t>')
        ans = ''
        if rel == '':
            continue
        if lang == 'es':
            ## for different forms of <Ser>
            if rel.split()[0] in ['es','ser', 'soy','están','estaba','fueron','estado']:
                if rel.split()[0] not in sentence:
                    new_rel = '<Ser> ' + ' '.join(rel.split()[1:])
                    rel = new_rel.strip()
        if lang == 'pt':
            ## for different forms of <Ser>
            if rel.split()[0] in ["é","<Ser>","ser",'sou','estão','estava','estamos','fui','sendo']:
                if rel.split()[0] not in sentence:
                    new_rel = '<Ser> ' + ' '.join(rel.split()[1:])
                    rel = new_rel.strip()

        if lang == 'zh':
            ## Handle for DESC relation in Chinese
            if rel.split()[0] in ['是']:
                if rel.split()[0] not in sentence:
                    new_rel = 'DESC' + ' '.join(rel.split()[1:])
                    rel = new_rel.strip()
        if arg1:
            ans += '<arg1> '
            ans += (arg1.strip() + ' ')
            ans += '</arg1> '
        ans += '<rel> '
        ans += (rel.strip() + ' ')
        ans += '</rel> '
        _arg2 = False
        if arg2:
            _arg2 = True
            ans += '<arg2> '
            ans += (arg2.strip() + ' ')
        if loc:
            if not _arg2:
                _arg2 = True
                ans += '<arg2> '
            ans += (loc.strip() + ' ')
        if time:
            if not _arg2:
                _arg2 = True
                ans += '<arg2> '
            ans += (time.strip() + ' ')
        if _arg2:
            ans += '</arg2>'
        ans = ans.strip()
        res.append(ans)
    res = list(set(res))
    # output_rel = []
    # stripped_output = []
    # output = []
    # for ans in res:
    #     clean_ans = []
    #     for word in ans.split():
    #         if word not in ['<arg1>', '</arg1>', '<arg2>', '</arg2>', '<rel>', '</rel>']:
    #             clean_ans.append(word)
    #     clean_ans = " ".join(clean_ans).strip()
    #     if clean_ans in stripped_output:
    #         ans_list = ans.split()
    #         index = stripped_output.index(clean_ans)
    #         if '<rel>' in ans_list and '</rel>' in ans_list:
    #             val = ans_list.index('</rel>') - ans_list.index('<rel>')
    #             if output_rel[index] < val:
    #                 output_rel[index] = val
    #                 output[index] = ans
    #     else:
    #         stripped_output.append(clean_ans)
    #         output.append(ans)
    #         ans_list = ans.split()
    #         if '<rel>' in ans_list and '</rel>' in ans_list:
    #             output_rel.append(ans_list.index('</rel>') - ans_list.index('<rel>'))
    #         else:
    #             output_rel.append(-1)
    # res = output
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser('clean input file')
    parser.add_argument('--inp_sentences', type=str, help='sentences')
    parser.add_argument('--inp_s2_processed', type=str, help='predictions')
    # parser.add_argument('--model_type', type=str, required=True, help='model type')
    # parser.add_argument('--data_type', type=str, required=True, help='data type')
    parser.add_argument('--lang', type=str, required=True, help='lang')
    parser.add_argument('--out_carb', type=str, help='output file')
    args = parser.parse_args()

    from os.path import expanduser
    home = expanduser("~")
    data_dir = home+"/moie_bucket/data"
  
    # args.fp1 = f"{data_dir}/carb/data/{args.lang}_test.input"
    # args.fp2 = f"{data_dir}/{args.lang}/{args.model_type}/{args.data_type}/test.predicted"
    # args.out = f"{data_dir}/{args.lang}/{args.model_type}/{args.data_type}/test.predicted.allennlp"
    with open(args.inp_sentences , 'r') as f1,\
        open(args.inp_s2_processed, 'r') as f2,\
            open(args.out_carb, 'w') as f3:
            sentences = f1.readlines()
            predictions = f2.readlines()
            assert len(sentences) == len(predictions), ipdb.set_trace()
            for ind in range(len(sentences)):
                out = allennlp(sentences[ind].strip().split(), predictions[ind].strip(), args.lang)
                for ind1 in range(len(out)):
                    # f4.write(sentences[ind].strip() + '\t' + out[ind1].strip() + '\n')
                    f3.write(sentences[ind].strip() + '\t' + out[ind1].strip() + '\t' + '1.0' + '\n')
    print('Output written to: ', args.out_carb)
