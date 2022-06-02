LANGUAGE=$1
MODEL_TYPE=$2
DATA_TYPE=$3
SPM=../sentencepiece/build/src/spm_encode
MODEL=../models/mbart.cc25.v2/sentence.bpe.model
if [[ "$MODEL_TYPE" == genoie ]]; then 
    DATA_FILE=./carb/data/${LANGUAGE}_test
    DEST_DIR=../models/${LANGUAGE}/${MODEL_TYPE}/test-bin
elif [[ "$MODEL_TYPE" == gen2oie_s1 ]]; then 
    DATA_FILE=./carb/data/${LANGUAGE}_test_s1
    DEST_DIR=../models/${LANGUAGE}/${MODEL_TYPE}/test-bin
elif [[ "$MODEL_TYPE" == gen2oie_s2 ]]; then 
    DATA_FILE=../models/${LANGUAGE}/${MODEL_TYPE}/${DATA_TYPE}-data/test
    DEST_DIR=../models/${LANGUAGE}/${MODEL_TYPE}/${DATA_TYPE}-bin
fi
SRC=input
TGT=target
${SPM} --model=${MODEL} < ${DATA_FILE}.${SRC} > ${DATA_FILE}.spm.${SRC} 
DICT=../models/mbart.cc25.v2/dict.txt
fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} --destdir ${DEST_DIR} --thresholdtgt 0    --thresholdsrc 0   --srcdict ${DICT}   --tgtdict ${DICT}   --workers 70 --testpref ${DATA_FILE}.spm --only-source
cp ${DICT} ${DEST_DIR}/dict.${TGT}.txt