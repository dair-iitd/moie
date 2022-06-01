LANGUAGE=$1
LANGUAGE_ID=$2
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
SPM=../sentencepiece/build/src/spm_encode
MODEL=../models/mbart.cc25.v2/sentence.bpe.model

### Inference using vanilla model
DATA=../models/${LANGUAGE}/mbart/vanilla-data
DICT=../models/mbart.cc25.v2/dict.txt
SRC=en_XX
TGT=${LANGUAGE_ID}

cp ../data/openie6/train.sentences ${DATA}/train.sentences.en_XX
cp ../data/openie6/train.extractions ${DATA}/train.extractions.en_XX

TEST=train.sentences
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC}
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT}  --destdir ../models/${LANGUAGE}/mbart/vanilla_sentences-bin  --thresholdtgt 0    --thresholdsrc 0   --srcdict ${DICT}   --tgtdict ${DICT}   --workers 70 --testpref ${DATA}/${TEST}.spm --only-source
cp ../models/mbart.cc25.v2/dict.txt ../models/${LANGUAGE}/mbart/vanilla_sentences-bin/dict.${SRC}.txt
cp ../models/mbart.cc25.v2/dict.txt ../models/${LANGUAGE}/mbart/vanilla_sentences-bin/dict.${TGT}.txt

TEST=train.extractions
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC}
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT}  --destdir ../models/${LANGUAGE}/mbart/vanilla_extractions-bin  --thresholdtgt 0    --thresholdsrc 0   --srcdict ${DICT}   --tgtdict ${DICT}   --workers 70 --testpref ${DATA}/${TEST}.spm --only-source
cp ../models/mbart.cc25.v2/dict.txt ../models/${LANGUAGE}/mbart/vanilla_extractions-bin/dict.${SRC}.txt
cp ../models/mbart.cc25.v2/dict.txt ../models/${LANGUAGE}/mbart/vanilla_extractions-bin/dict.${TGT}.txt

CUDA_VISIBLE_DEVICES=1 fairseq-generate ../models/${LANGUAGE}/mbart/vanilla_sentences-bin --path ../models/${LANGUAGE}/mbart/vanilla-checkpoints/checkpoint_best.pt  --task translation_from_pretrained_bart --gen-subset test -t ${LANGUAGE_ID} -s ${SRC} --sacrebleu --remove-bpe 'sentencepiece' --langs $langs --max-sentences 8 --max-source-positions 512 > ${DATA}/sentences_output
cat ${DATA}/sentences_output | grep -P "^H" | sort -V | cut -f 3- > ../data/${LANGUAGE}/mbart/train.sentences

CUDA_VISIBLE_DEVICES=2 fairseq-generate ../models/${LANGUAGE}/mbart/vanilla_extractions-bin --path ../models/${LANGUAGE}/mbart/vanilla-checkpoints/checkpoint_best.pt  --task translation_from_pretrained_bart --gen-subset test -t ${LANGUAGE_ID} -s ${SRC} --sacrebleu --remove-bpe 'sentencepiece' --langs $langs --max-sentences 8 --max-source-positions 512 > ${DATA}/extractions_output
cat ${DATA}/extractions_output | grep -P "^H" | sort -V | cut -f 3- > ../data/${LANGUAGE}/mbart/train.extractions

### Generate inputs for consistent inference
python aligner.py --inp1 ../data/openie6/train.sentences --inp2 ../data/${LANGUAGE}/mbart/train.sentences --output_file ../data/${LANGUAGE}/mbart/consistent/word_aligned.sentences --alignment_type translation_sentence --lang ${LANGUAGE} --model_name_or_path ../models/${LANGUAGE}/model_without_co/
python aligner.py --inp1 ../data/openie6/train.sentences_labels_aux  --inp2 ../data/${LANGUAGE}/mbart/train.sentences_repeated  --output_file ../data/${LANGUAGE}/mbart/consistent/word_aligned.extractions --alignment_type translation_extraction --lang ${LANGUAGE} --model_name_or_path ../models/${LANGUAGE}/model_without_co/


### Inference using consistent model
DATA=../models/${LANGUAGE}/mbart/consistent10-data
DICT=../models/mbart.cc25.v2/dict.txt
SRC=input
TGT=${LANGUAGE_ID}

cp ../data/${LANGUAGE}/mbart/consistent/word_aligned.extractions ${DATA}/word_aligned.extractions.input
cp ../data/${LANGUAGE}/mbart/consistent/word_aligned.sentences ${DATA}/word_aligned.sentences.input

TEST=word_aligned.sentences
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC}
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT}  --destdir ../models/${LANGUAGE}/mbart/consistent10_sentences-bin  --thresholdtgt 0    --thresholdsrc 0   --srcdict ${DICT}   --tgtdict ${DICT}   --workers 70 --testpref ${DATA}/${TEST}.spm --only-source
cp ../models/mbart.cc25.v2/dict.txt ../models/${LANGUAGE}/mbart/consistent10_sentences-bin/dict.${LANGUAGE_ID}.txt

TEST=word_aligned.extractions
${SPM} --model=${MODEL} < ${DATA}/${TEST}.${SRC} > ${DATA}/${TEST}.spm.${SRC}
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT}  --destdir ../models/${LANGUAGE}/mbart/consistent10_extractions-bin  --thresholdtgt 0    --thresholdsrc 0   --srcdict ${DICT}   --tgtdict ${DICT}   --workers 70 --testpref ${DATA}/${TEST}.spm --only-source
cp ../models/mbart.cc25.v2/dict.txt ../models/${LANGUAGE}/mbart/consistent10_extractions-bin/dict.${LANGUAGE_ID}.txt

fairseq-generate ../models/${LANGUAGE}/mbart/consistent10_sentences-bin --path ../models/${LANGUAGE}/mbart/consistent10-checkpoints/checkpoint_best.pt  --task translation_from_pretrained_bart --gen-subset test -t ${LANGUAGE_ID} -s input --sacrebleu --remove-bpe 'sentencepiece' --langs $langs --max-sentences 32 --max-source-positions 512 > ${DATA}/sentences_output
cat ${DATA}/sentences_output | grep -P "^H" | sort -V | cut -f 3- > ../data/${LANGUAGE}/mbart10/consistent/train.sentences


fairseq-generate ../models/${LANGUAGE}/mbart/consistent10_extractions-bin --path ../models/${LANGUAGE}/mbart/consistent10-checkpoints/checkpoint_best.pt  --task translation_from_pretrained_bart --gen-subset test -t ${LANGUAGE_ID} -s input --sacrebleu --remove-bpe 'sentencepiece' --langs $langs --max-sentences 32 --max-source-positions 512 > ${DATA}/extractions_output
cat ${DATA}/extractions_output | grep -P "^H" | sort -V | cut -f 3- > ../data/${LANGUAGE}/mbart/consistent/train.extractions

