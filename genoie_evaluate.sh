LANGUAGE=$1
DATA_TYPE=$2
CHECKPOINT=$3
TPU=$4

bash predict.sh ${LANGUAGE}_genoie_${DATA_TYPE} ${TPU} ${CHECKPOINT} carb/data/${LANGUAGE}_test.input ${LANGUAGE}/genoie/${DATA_TYPE}/test.predicted

python create_allennlp.py --model_type genoie --data_type ${DATA_TYPE} --lang ${LANGUAGE}

# python carb/carb.py --gold carb/data/gold/${LANGUAGE}_test.tsv --out /dev/null --allennlp /home/muqeeth101/moie_bucket/data/${LANGUAGE}/genoie/${DATA_TYPE}/test.predicted.allennlp

cd $HOME/moie_bucket/data/${LANGUAGE}/genoie/${DATA_TYPE}/
cut -f 1 test.predicted.allennlp > scores.input
cut -f 2 test.predicted.allennlp > scores.output
cp scores.output scores.output.corr_tags
sed -i 's/<arg1>/<a1>/' scores.output.corr_tags
sed -i 's/<\/arg1>/<\/a1>/' scores.output.corr_tags
sed -i 's/<arg2>/<a2>/' scores.output.corr_tags
sed -i 's/<\/arg2>/<\/a2>/' scores.output.corr_tags
sed -i 's/<rel>/<r>/' scores.output.corr_tags
sed -i 's/<\/rel>/<\/r>/' scores.output.corr_tags

cd -

bash score.sh ${LANGUAGE}_genoie_${DATA_TYPE} ${TPU} ${CHECKPOINT} ${LANGUAGE}/genoie/${DATA_TYPE}/scores.input ${LANGUAGE}/genoie/${DATA_TYPE}/scores.output.corr_tags

cd $HOME//moie_bucket/data/${LANGUAGE}/genoie/${DATA_TYPE}/
paste scores.input scores.output scores.output.corr_tags.scores > final.allennlp.$CHECKPOINT

cd -
python carb/carb.py --gold carb/data/gold/${LANGUAGE}_test.tsv --out /dev/null --allennlp $HOME/moie_bucket/data/${LANGUAGE}/genoie/${DATA_TYPE}/final.allennlp.${CHECKPOINT}

