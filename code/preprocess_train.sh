LANGUAGE=$1
MODEL_TYPE=$2
DATA_TYPE=$3
SPM=../sentencepiece/build/src/spm_encode
MODEL=../models/mbart.cc25.v2/sentence.bpe.model
DATA=../models/${LANGUAGE}/${MODEL_TYPE}/${DATA_TYPE}-data
TRAIN=train
VALID=valid
SRC=input
TGT=target
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${SRC} > ${DATA}/${TRAIN}.spm.${SRC} 
${SPM} --model=${MODEL} < ${DATA}/${TRAIN}.${TGT} > ${DATA}/${TRAIN}.spm.${TGT}
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${SRC} > ${DATA}/${VALID}.spm.${SRC}
${SPM} --model=${MODEL} < ${DATA}/${VALID}.${TGT} > ${DATA}/${VALID}.spm.${TGT}
DICT=../models/mbart.cc25.v2/dict.txt
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} --destdir ../models/${LANGUAGE}/${MODEL_TYPE}/${DATA_TYPE}-bin --thresholdtgt 0    --thresholdsrc 0   --srcdict ${DICT}   --tgtdict ${DICT}   --workers 70 --trainpref ${DATA}/${TRAIN}.spm --validpref ${DATA}/${VALID}.spm

# bash preprocess_train.sh hi genoie clp
# bash preprocess_train.sh hi genoie translate_clp
# bash preprocess_train.sh hi genoie ctranslate_clp
# bash preprocess_train.sh hi gen2oie_s1 clp
# bash preprocess_train.sh hi gen2oie_s1 translate_clp
# bash preprocess_train.sh hi gen2oie_s1 ctranslate_clp
# bash preprocess_train.sh hi gen2oie_s2 clp
# bash preprocess_train.sh hi gen2oie_s2 translate_clp
# bash preprocess_train.sh hi gen2oie_s2 ctranslate_clp

# bash preprocess_train.sh es genoie ctranslate_clp
# bash preprocess_train.sh es gen2oie_s1 ctranslate_clp
# bash preprocess_train.sh es gen2oie_s2 ctranslate_clp

# Ablations
# bash preprocess_train.sh es gen2oie_s1 ctranslate_clp_rand_sort
# bash preprocess_train.sh es gen2oie_s1 ctranslate_clp_no_verb_removal
# bash preprocess_train.sh es gen2oie_s1 ctranslate_clp_no_pos_tags

# bash preprocess_train.sh zh gen2oie_s1 ctranslate_clp_rand_sort
# bash preprocess_train.sh zh gen2oie_s1 ctranslate_clp_no_verb_removal
# bash preprocess_train.sh zh gen2oie_s1 ctranslate_clp_no_pos_tags

# bash preprocess_train.sh hi gen2oie_s1 ctranslate_clp_rand_sort
# bash preprocess_train.sh hi gen2oie_s1 ctranslate_clp_no_verb_removal
# bash preprocess_train.sh hi gen2oie_s1 ctranslate_clp_no_pos_tags