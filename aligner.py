import torch
import transformers
import itertools
from transformers import AutoModel, AutoTokenizer
import ipdb
import pickle
import argparse
import os

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from  phrase_extract import phrase_extraction
import stanza
import string
from tqdm import tqdm
import random
import multiprocessing

from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
import time
from nltk.translate.bleu_score import sentence_bleu

def set_seed(args):
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


punct = string.punctuation
auxillary_dict = {'es':{'is':'es','of':'de','from':'de'},\
                'zh':{'is':'是','of':'的','from':'的'},\
                'hi':{'is':'है','of':'का','from':'से'},\
                'pt':{'is':'é','of':'de','from':'de'}, \
                'te':{'is':'ఉంది','of':'యొక్క','from':'నుండి'}}

def get_spans(sent, label):
    label_dict = {'ARG1': 1, 'REL': 2, 'ARG2': 3, 'LOC': 4, 'TIME':5}
    ind = 0
    spans = []
    while ind < len(label):
        if label[ind] != 'NONE':
            start_index = ind
            end_index = ind
            prev_label = label[ind]
            ind+=1
            while ind < len(label) and (label[ind] == prev_label):
                end_index = ind
                ind+=1
            spans.append(((start_index, end_index+1), " ".join(sent[start_index:end_index+1]).strip(), prev_label))
        else:
            ind+=1
    return spans


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, args):
        assert os.path.isfile(args.inp1) and os.path.isfile(args.inp2)

        print('Loading the dataset...')
        self.examples = []
        with open(args.inp1) as f1,\
            open(args.inp2) as f2:
            data1 = f1.readlines()
            data2 = f2.readlines()
            if args.debug:
                data1 = data1[:200]
                data2 = data2[:200]
                args.output_file = args.output_file + '.debug'

            assert len(data1) == len(data2)

            for idx, line1 in tqdm(enumerate(data1)):
                if len(line1) == 0 or line1.isspace():
                    line1 = 'none'
                    # raise ValueError(f'Line {idx+1} is not in the correct format!')

                line2 = data2[idx]
                if len(line2) == 0 or line2.isspace():
                    line2 = 'none'
                    # raise ValueError(f'Line {idx+1} is not in the correct format!')

                if args.alignment_type == 'translation_sentence':
                    src = line1.strip()
                    tgt = line2.strip()
                    label = ['ARG1' for _ in range(len(src.split()))]
                else:
                    src, label = line1.split('|||')
                    src = src.strip()
                    label = label.strip()
                    tgt = line2.strip()
                    label = label.split()

                if src.rstrip() == '' or tgt.rstrip() == '':
                    raise ValueError(f'Line {idx+1} is not in the correct format!')

                sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
                assert (len(label) == len(sent_src)) or (len(label) == len(sent_src)+3), ipdb.set_trace()
                
                if 'translation' in args.alignment_type:
                    label_index = {}
                else:
                    label_index = get_spans(sent_src, label)
                    # if 'REL' in label:
                        # rel_index = label.index('REL')
                        # if sent_src[rel_index] in ['to', 'for', 'of', 'in', 'on', 'from', 'at', 'by', 'with', 'without', 'into']:
                        #     label_index = []
                        # else:
                        #     label_index = get_spans(sent_src,label)
                    # else:
                        # label_index = get_spans(sent_src, label)
                token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
                wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
                ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
                bpe2word_map_src = []
                for i, word_list in enumerate(token_src):
                    bpe2word_map_src += [i for x in word_list]
                bpe2word_map_tgt = []
                for i, word_list in enumerate(token_tgt):
                    bpe2word_map_tgt += [i for x in word_list]


                self.examples.append( (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, label,label_index, sent_src, sent_tgt) )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def word_align(args, model, tokenizer,nlp):

    def collate(examples):
        ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, label, label_index, sent_src, sent_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, label, label_index, sent_src, sent_tgt
    def clp_sentence(phrases, label_index, sent_tgt, word_aligns):
        tgt_label = ['NONE' for _ in range(len(sent_tgt))]
        word_aligns_dict = {}
        for key in word_aligns:
            try:
                tmp = word_aligns_dict[key[0]]  
                word_aligns_dict[key[0]].append(key[1])
            except:
                word_aligns_dict[key[0]] = [key[1]]      
        for ind, l in enumerate(label_index):
            max_score, span = -1, -1
            for phr in phrases:
                phrase_len = min(len(l[1].split()), len(phr[2].split()))
                score = 0
                if phrase_len >3:
                    weight = (0.25, 0.25, 0.25, 0.25)
                    score = round(sentence_bleu([l[1].split()], phr[2].split(), weights = weight),3)
                if score ==0 or phrase_len == 3:
                    weight = (1/3, 1/3, 1/3)
                    score = round(sentence_bleu([l[1].split()], phr[2].split(), weights = weight),3)
                if score==0 or phrase_len == 2:
                    weight = (0.5, 0.5)
                    score = round(sentence_bleu([l[1].split()], phr[2].split(), weights = weight),3)
                if score==0 or phrase_len == 1:
                    weight = (1,)
                    score = round(sentence_bleu([l[1].split()], phr[2].split(), weights = weight),3)

                if score > max_score:
                    max_score = score
                    span = phr
                elif score == max_score:
                    if args.lang == 'hi':
                        if (phr[1][1] - phr[1][0]) > (span[1][1] - span[1][0]):
                            span = phr
                    else:
                        if (phr[1][1] - phr[1][0]) < (span[1][1] - span[1][0]):
                            span = phr 
            if max_score == 0:
                for i in range(l[0][0],l[0][1]):
                    try:
                        res = word_aligns_dict[i]
                        for index in res:
                            # if tgt_label[index] != 'NONE':
                            #     ipdb.set_trace()
                            tgt_label[index] = l[2]
                    except:
                        pass
            else:
                for i in range(span[1][0], span[1][1]):
                    # if tgt_label[i] != 'NONE':
                    #     ipdb.set_trace()
                    tgt_label[i] = l[2]
        return " ".join(tgt_label).strip()

    def clp_extraction(phrases, label_index, sent_tgt, word_aligns, tgt):
        def best_match(x, phrases):
            keys = list(phrases.keys())
            max_overlap = 0
            max_overlap_index = -1
            for ind, key in enumerate(keys):
                if key[0]<=x[0] and key[1]>=x[1]:
                    num = len(set([i for i in range(key[0],key[1])]).intersection(set([i for i in range(x[0],x[1])])))
                    den = len(set([i for i in range(key[0],key[1])]).union(set([i for i in range(x[0],x[1])])))
                    overlap = num/den
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_overlap_index = ind
                    elif overlap == max_overlap and overlap != 0:
                        if keys[ind][0]<keys[max_overlap_index][0]:
                            max_overlap_index = ind

            if max_overlap_index == -1:
                ipdb.set_trace()
            return keys[max_overlap_index], max_overlap
        tgt_index = []
        min_tgt_index = []
        tgt_map = []

        tgt_index_min = {}
        tgt_index_max = {}
        for ind, l in enumerate(label_index):
            min_span_len, min_span, max_span_len, max_span = 10000, '',-1,''
            match, overlap = best_match(l[0], phrases)
            for phr in phrases[match]:
                if (phr[1][1] - phr[1][0]) > max_span_len:
                    max_span_len = phr[1][1] - phr[1][0]
                    max_span = phr[1]
                if (phr[1][1] - phr[1][0]) < min_span_len:
                    min_span_len = phr[1][1] - phr[1][0]
                    min_span = phr[1]
            if overlap ==1:
                tgt_map.append(True)
            else:
                tgt_map.append(False)
            if min_span == '':
                # ipdb.set_trace()
                continue
            tgt_index.append((max_span, l[2])) 
            min_tgt_index.append((min_span, l[2]))
            try:
                tmp = tgt_index_min[l[2]]
                tgt_index_min[l[2]].append(min_span)
                tgt_index_max[l[2]].append(max_span)
            except:
                tgt_index_min[l[2]] = [min_span]
                tgt_index_max[l[2]] = [max_span]
        tgt_label = ['NONE' for _ in range(len(sent_tgt))]

        unmapped_words = []
        unmapped_prev_labels = {}
        for key in tgt_index_max:
            for index,sp in enumerate(tgt_index_max[key]):
                for ind in range(sp[0],sp[1]):
                    if tgt_label[ind] == 'NONE':
                        # FIXME: remove it
                        if key == 'REL' and tgt[ind]['upos'] == 'PUNCT':
                            tgt_label[ind] = 'NONE'
                        else:
                            tgt_label[ind] = key
                    elif key != tgt_label[ind]:
                        # FIXME: works for two overlapping phrases
                        tgt_label[ind] = 'NONE'
                        unmapped_words.append(ind)
                        unmapped_prev_labels[ind] = tgt_label[ind]
        prev_tgt_label = tgt_label.copy()
        unmapped_words = list(set(unmapped_words))
        prev_tgt_label = tgt_label.copy()

        def children(tgt, w):
            res = []
            for d in tgt:
                if d['head']-1 == w:
                    res.append(d['id']-1)
            return res
        while True:
            # FIXME: Use the help of unmapped prev labels
            for w in unmapped_words:
                if len([c for c in children(tgt,w)]) == 0:
                    tgt_label[w] = tgt_label[tgt[w]['head']-1]
                if tgt[w]['head'] == 0 and len([c for c in children(tgt,w)]) == 1:
                    childs = [c for c in children(tgt,w)]
                    tgt_label[w] = tgt_label[childs[0]]
                # def check(w):
                #     for sp in tgt_index_max['REL']:
                #         if w>=sp[0] and w<=sp[1]:
                #             return True
                #     return False
                # if tgt[w]['upos'] in ['VERB', 'AUX'] and check(w):
                if tgt[w]['upos'] in ['VERB', 'AUX']:
                    tgt_label[w] = 'REL'
            if tgt_label == prev_tgt_label:
                break
            prev_tgt_label = tgt_label.copy()

        for ind1, l in enumerate(min_tgt_index):
            if tgt_map[ind1] == True:
                for ind2 in range(min_tgt_index[ind1][0][0], min_tgt_index[ind1][0][1]):
                    tgt_label[ind2] = min_tgt_index[ind1][1]
                    if ind2 in unmapped_words:
                        index2 = unmapped_words.index(ind2)
                        unmapped_words.remove(ind2)
        while True:
            for w in unmapped_words:
                if len([c for c in children(tgt,w)]) == 0:
                    tgt_label[w] = tgt_label[tgt[w]['head']-1]
                if tgt[w]['head'] == 0 and len([c for c in children(tgt,w)]) == 1:
                    childs = [c for c in children(tgt,w)]
                    tgt_label[w] = tgt_label[childs[0]]
                # def check(w):
                #     for sp in tgt_index_max['REL']:
                #         if w>=sp[0] and w<=sp[1]:
                #             return True
                #     return False
                # if tgt[w]['upos'] in ['VERB', 'AUX'] and check(w):
                if tgt[w]['upos'] in ['VERB', 'AUX']:
                    tgt_label[w] = 'REL'
            if tgt_label == prev_tgt_label:
                break
            prev_tgt_label = tgt_label.copy()

        for ind, w in enumerate(unmapped_words):
            if tgt_label[w] == 'NONE':
                tgt_label[w] = unmapped_prev_labels[w]

        if sent_tgt[-1] in punct:
            tgt_label[-1] = 'NONE'
        tgt_label = " ".join(tgt_label)

        return tgt_label
    def make_dict(phrases):
        res = {}
        for phr in phrases:
            try:
                tmp = res[phr[0]]
                res[phr[0]].append((phr[2], phr[1], phr[3]))
            except:
                res[phr[0]] = [(phr[2], phr[1], phr[3])]
        return res
    
    def translation_alignment(tgt, word_aligns, sent_src, sent_tgt, label, lang, base_d):
        word_aligns_dict = {}
        for word_tuple in word_aligns:
            try:
                tmp = word_aligns_dict[word_tuple[0]]
                word_aligns_dict[word_tuple[0]].append(word_tuple[1])
            except:
                word_aligns_dict[word_tuple[0]] = [word_tuple[1]]
        constrained_train = []
        new_tag = []
        for ind1 in range(len(sent_src)):
            if label[ind1] != 'NONE':
                new_tag.append(label[ind1])
                if ind1 in word_aligns_dict:
                    tmp = word_aligns_dict[ind1]
                    tgt_word = []
                    for ind2 in tmp:
                        if tgt:
                            assert tgt[ind2]['text'] == sent_tgt[ind2], ipdb.set_trace()
                            if tgt[ind2]['text'] not in punct:
                                # p = random.random()
                                # if p<=0.5:
                                #     tgt_word.append(tgt[ind2]['lemma'])
                                # else:
                                #     tgt_word.append(sent_tgt[ind2])
                                if base_d:
                                    tgt_word.append(base_d[tgt[ind2]['text']])
                                else:
                                    tgt_word.append(tgt[ind2]['lemma'])
                            else:
                                tgt_word.append(tgt[ind2]['text'])
                        else:
                            tgt_word.append(sent_tgt[ind2])
                    constrained_train.append('# ' + sent_src[ind1].strip() + ' ## ' + " ".join(tgt_word).strip() + ' #')
                else:
                    constrained_train.append('# ' + sent_src[ind1].strip() + ' #')
        if len(label) == len(sent_src) + 3:
            unused = -1
            if label[-1] == 'REL':
                unused = 3
            elif label[-2] == 'REL':
                unused = 2
            elif label[-3] == 'REL':
                unused = 1 
            if 'REL' in new_tag:
                index1 = new_tag.index('REL')
                index2 = len(new_tag) -1 -new_tag[::-1].index('REL')
                assert index1 <= index2
                if unused == 1:
                    constrained_train = constrained_train[:index1] + [f"# is ## {auxillary_dict[lang]['is']} #"] + constrained_train[index1:]
                if unused == 2:
                    constrained_train = constrained_train[:index1] + [f"# is ## {auxillary_dict[lang]['is']} #"] + constrained_train[index1:index2+1] + [f"# of ## {auxillary_dict[lang]['of']} #"] + constrained_train[index2+1:]
                if unused == 3:
                    constrained_train = constrained_train[:index1] + [f"# is ## {auxillary_dict[lang]['is']} #"] + constrained_train[index1:index2+1] + [f"# from ## {auxillary_dict[lang]['from']} #"] + constrained_train[index2+1:]
            else:
                index1 = len(new_tag) - 1 - new_tag[::-1].index('ARG1')
                index2 = len(new_tag) - 1 - new_tag[::-1].index('ARG2')
                index = min(index1, index2)
                assert unused == 1
                constrained_train = constrained_train[:index+1] + [f"# is ## {auxillary_dict[lang]['is']} #"] + constrained_train[index+1:]
        return " ".join(constrained_train).strip() + '\n'

    dataset = LineByLineTextDataset(tokenizer, args)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate
    )

    model.to(args.device)
    model.eval()
    tqdm_iterator = trange(dataset.__len__(), desc="Extracting")
    all_sent_src, all_sent_tgt, all_word_aligns, all_label_index,all_tgt,all_label = [],[],[],[],[],[]
    with open(args.output_file, 'w') as writer:
        for batch in dataloader:
            with torch.no_grad():
                ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, label, label_index, sent_src, sent_tgt = batch
                word_aligns_list = model.get_aligned_word(ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, args.device, None, None, test=True)
                for ind, word_aligns in enumerate(word_aligns_list):
                    all_sent_src.append(sent_src[ind])
                    all_sent_tgt.append(sent_tgt[ind])
                    all_word_aligns.append(word_aligns)
                    all_label_index.append(label_index[ind])
                    all_label.append(label[ind])
                tqdm_iterator.update(len(ids_src))
        if args.alignment_type == "clp_extraction" or args.lemma:
            start_time = time.time()
            for sentence in nlp(all_sent_tgt).sentences:
                all_tgt.append(sentence.to_dict())
            end_time = time.time()
            print('time taken for stanza is ', end_time-start_time)

        def parallel_project(sent_src_list, sent_tgt_list, label_index_list, word_aligns_list, worker_id,tgt_list, label_list, lang, base_d):
            output = []
            for ind in tqdm(range(len(sent_src_list))):
                sent_src = sent_src_list[ind]
                sent_tgt = sent_tgt_list[ind]
                label_index = label_index_list[ind]
                word_aligns = word_aligns_list[ind]
                label = label_list[ind]
                if args.alignment_type == 'clp_sentence':
                    phrases = list(phrase_extraction(" ".join(sent_src), " ".join(sent_tgt), list(word_aligns)))
                    tgt_label = clp_sentence(phrases, label_index, sent_tgt, word_aligns)
                    output.append(" ".join(sent_tgt) + ' ||| ' + tgt_label + '\n')
                elif args.alignment_type == 'clp_extraction':
                    phrases = list(phrase_extraction(" ".join(sent_src), " ".join(sent_tgt), list(word_aligns)))
                    phrases = make_dict(phrases)
                    tgt = tgt_list[ind]
                    tgt_label = clp_extraction(phrases, label_index, sent_tgt, word_aligns,tgt)
                    output.append(" ".join(sent_tgt) + ' ||| ' + tgt_label + '\n')
                elif 'translation' in args.alignment_type:
                    if args.lemma:
                        tgt = tgt_list[ind]
                        output.append(translation_alignment(tgt, word_aligns,sent_src,sent_tgt,label, lang, base_d))
                    else:
                        output.append(translation_alignment(None, word_aligns,sent_src,sent_tgt,label,lang, base_d))
                
            with open(args.output_file +'_' + str(worker_id), 'w') as f:
                f.write("".join(output))
        workers = 10
        each_load = int(len(all_sent_src)/workers)+1
        processes = []

        base_d = None
        if args.lang == 'te':
            base_d = pickle.load(open('../data/te/mbart/train.sentences.base.pkl','rb'))

        for i in range(workers):
            processes.append(multiprocessing.Process(target=parallel_project, args=(all_sent_src[i*each_load:(i+1)*each_load],all_sent_tgt[i*each_load:(i+1)*each_load],\
                all_label_index[i*each_load:(i+1)*each_load], all_word_aligns[i*each_load:(i+1)*each_load], i,all_tgt[i*each_load:(i+1)*each_load],\
                    all_label[i*each_load:(i+1)*each_load],args.lang, base_d)))
        for i in range(workers):
            processes[i].start()
        for i in range(workers):
            processes[i].join()
        final_output = []
        for i in range(workers):
            with open(args.output_file + '_' + str(i), 'r') as f:
                worker_data = f.readlines()
            final_output.extend(worker_data)
            os.remove(args.output_file +'_'+str(i))
        writer.write("".join(final_output))
        
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--inp1", default=None, type=str, required=True, help="The input data file (a text file)."
    )
    parser.add_argument(
        "--inp2", default=None, type=str, required=True, help="The input data file (a text file)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--softmax_threshold", type=float, default=0.001
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--align_layer", type=int, default=8, help="layer for alignment extraction")
    parser.add_argument(
        "--extraction", default='softmax', type=str, help='softmax or entmax15'
    )
    parser.add_argument(
        "--alignment_type", type=str, help='clp_sentence or clp_extraction or translation_sentence or translation_extraction'
    )
    parser.add_argument(
        "--lang", type=str, help='es or zh or hi'
    )
    parser.add_argument("--lemma", action = 'store_true')
    parser.add_argument("--debug", action = 'store_true')
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    set_seed(args)
    config_class, model_class = BertConfig, BertForMaskedLM
    config = config_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config
    )

    # if args.lang == 'te' or args.lang == 'hi':
    #     args.lemma = True
    proc_lang = args.lang

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    nlp = stanza.Pipeline(proc_lang, tokenize_pretokenized= True, tokenize_no_ssplit=True, tokenize_batch_size=4096)
    word_align(args, model, tokenizer,nlp)
if __name__ == "__main__":
    main()
