LANGUAGE=$1
DATA_TYPE=$2
CHECKPOINT1=$3
CHECKPOINT2=$4
TPU=$5

echo $CHECKPOINT1 $CHECKPOINT2

bash predict.sh ${LANGUAGE}_gen2oie_s1_${DATA_TYPE} ${TPU} ${CHECKPOINT1} carb/data/${LANGUAGE}_test.input ${LANGUAGE}_gen2oie_s1_${DATA_TYPE}_test.predicted

# python make_test_s2.py --lang ${LANGUAGE} --data_type ${DATA_TYPE} 
# bash predict.sh ${LANGUAGE}_gen2oie_s2_${DATA_TYPE} ${TPU} ${CHECKPOINT2} ${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/test.input ${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/test.pre_predicted

# python postprocess_test_s2.py --lang ${LANGUAGE} --data_type ${DATA_TYPE} 

# python create_allennlp.py --model_type gen2oie_s2 --data_type ${DATA_TYPE} --lang ${LANGUAGE}

# #python carb/carb.py --gold carb/data/gold/${LANGUAGE}_test.tsv --out /dev/null --allennlp /home/muqeeth101/moie_bucket/data/${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/test.predicted.allennlp

# ## Compute the scores according to the gen2oie model
# cd $HOME/moie_bucket/data/${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/
# cut -f 1 test.predicted.allennlp > scores.input
# cut -f 2 test.predicted.allennlp > scores.output
# cp scores.output scores.output.corr_tags
# sed -i 's/<arg1>/<a1>/' scores.output.corr_tags
# sed -i 's/<\/arg1>/<\/a1>/' scores.output.corr_tags
# sed -i 's/<arg2>/<a2>/' scores.output.corr_tags
# sed -i 's/<\/arg2>/<\/a2>/' scores.output.corr_tags
# sed -i 's/<rel>/<r>/' scores.output.corr_tags
# sed -i 's/<\/rel>/<\/r>/' scores.output.corr_tags


# cd -
# bash score.sh ${LANGUAGE}_gen2oie_s2_${DATA_TYPE} ${TPU} ${CHECKPOINT2} ${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/scores.input ${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/scores.output.corr_tags

# cd $HOME/moie_bucket/data/${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/
# paste scores.input scores.output scores.output.corr_tags.scores > "final.allennlp."$CHECKPOINT1"_"$CHECKPOINT2

# cd -
# python carb/carb.py --gold carb/data/gold/${LANGUAGE}_test.tsv --out /dev/null --allennlp ~/moie_bucket/data/${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/"final.allennlp."$CHECKPOINT1"_"$CHECKPOINT2 > ~/moie_bucket/data/${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/${CHECKPOINT1}_${CHECKPOINT2}.score
# cat ~/moie_bucket/data/${LANGUAGE}/gen2oie_s2/${DATA_TYPE}/${CHECKPOINT1}_${CHECKPOINT2}.score

