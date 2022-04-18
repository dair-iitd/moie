import argparse

parser = argparse.ArgumentParser(description='creating test data for label_rescoring using allennlp')
parser.add_argument("--inp", help = "path to test file in allennlp format")
parser.add_argument("--out", help = "path to save extaction_sentences with its labels")
parser.add_argument("--openie6_predictions", default=False, help = "path to gen2oie output predictions.txt.oie")
args=parser.parse_args()

file = open(args.inp, 'r') #'/home/shubham/openie6/data/en_gen2oie_test.allennlp'
lines = file.readlines()
print("number of extractions", len(lines))

lines = [l.replace('\t', ' ') for l in lines]
lines = [l.split(' ') for l in lines]

ext_to_orig_sen = {}
ext_sen_lbls=[]
for line_no, l in enumerate(lines):
    new_line = []
    if('<arg1>' in l):
        new_line += l[l.index('<arg1>')+1:l.index('</arg1>')]
    if('<rel>' in l):
        new_line += l[l.index('<rel>')+1:l.index('</rel>')]
    if('<arg2>' in l):
        new_line += l[l.index('<arg2>')+1:l.index('</arg2>')]

    # empty extraction
    if(len(new_line)==0):
        #print(line_no, l, sentences[line_no])
        continue

    ext_sen = ' '.join(new_line)+'\n'
    ext_sen_lbls.append(ext_sen)

    ext_sen_label = []
    if('<arg1>' in l):
        ext_sen_label += ['ARG1']*(l.index('</arg1>') - l.index('<arg1>') - 1)
    if('<rel>' in l):
        ext_sen_label += ['REL']*(l.index('</rel>') - l.index('<rel>') - 1)
    if('<arg2>' in l):
        ext_sen_label += ['ARG2']*(l.index('</arg2>') - l.index('<arg2>') - 1)

    ext_sen_label = ' '.join(ext_sen_label)+'\n'
    ext_sen_lbls.append(ext_sen_label)
    # format requires empty line separated data points
    ext_sen_lbls.append('\n')
    
    if('<arg1>' in l):
        ext_to_orig_sen[ext_sen] = ' '.join(l[0:l.index('<arg1>')])+'\n'
    elif('<rel>' in l):
        ext_to_orig_sen[ext_sen] = ' '.join(l[0:l.index('<rel>')])+'\n'

    # # multiple same extractions
    # if(ext_sen in ext_to_orig_sen.keys()):
    #     print(ext_sen)
    #     print(ext_sen_label)
    #     print(' '.join(l[0:l.index('<arg1>')]))
    #     print(ext_to_orig_sen[ext_sen])
    #     break
    # else:
    #     ext_to_orig_sen[ext_sen] = ' '.join(l[0:l.index('<arg1>')])+'\n'+'<SPCL>'+ext_sen_label

if args.openie6_predictions == False:
    write_file = open(args.out, 'w') #'/home/shubham/openie6/data/en_gen2oie_test.allennlp.ext_sen_lbls'
    write_file.writelines(ext_sen_lbls)
    write_file.close()


if args.openie6_predictions != False:
    prediction = open(args.openie6_predictions+".oie", 'r')
    prediction_lines = prediction.readlines()
    for i in range(len(prediction_lines)):
        if(i==0):
            prediction_lines[i] = ext_to_orig_sen[prediction_lines[i]]
        else:
            if(prediction_lines[i-1]=='\n'):
                prediction_lines[i] = ext_to_orig_sen[prediction_lines[i]]
    write_file = open(args.openie6_predictions+"2.oie", 'w')
    write_file.writelines(prediction_lines)
    write_file.close()