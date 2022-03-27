LANGUAGE=$1
MODEL_S1=$2
MODEL_S2=$3
TPU=$4
INPUT=$5
OUTPUT=$6

bash predict.sh ${MODEL_S1} ${TPU} ${INPUT} ${OUTPUT}.s1_predicted
python make_test_s2.py --inp_s1_predictions ${OUTPUT}.s1_predicted --inp_sentences ${INPUT} --out_s2_input ${OUTPUT}  

bash predict.sh ${MODEL_S2} ${TPU} ${OUTPUT}.s2_input ${OUTPUT}.s2_output
python postprocess_test_s2.py --inp_s2_count ${OUTPUT}.s2_count --inp_s2_output ${OUTPUT}.s2_output --out_s2_processed ${OUTPUT}.s2_processed 
python create_carb.py  --inp_sentences ${INPUT} --inp_s2_processed ${OUTPUT}.s2_processed --out_carb ${OUTPUT}.pre_score_carb --lang ${LANGUAGE}

## Compute the scores according to the gen2oie model
cut -f 1 ${OUTPUT}.pre_score_carb > ${OUTPUT}.scores_input
cut -f 2 ${OUTPUT}.pre_score_carb > ${OUTPUT}.scores_output
cp ${OUTPUT}.scores_output ${OUTPUT}.scores_output_corr_tags
sed -i 's/<arg1>/<a1>/' ${OUTPUT}.scores_output_corr_tags
sed -i 's/<\/arg1>/<\/a1>/' ${OUTPUT}.scores_output_corr_tags
sed -i 's/<arg2>/<a2>/' ${OUTPUT}.scores_output_corr_tags
sed -i 's/<\/arg2>/<\/a2>/' ${OUTPUT}.scores_output_corr_tags
sed -i 's/<rel>/<r>/' ${OUTPUT}.scores_output_corr_tags
sed -i 's/<\/rel>/<\/r>/' ${OUTPUT}.scores_output_corr_tags

bash score.sh ${MODEL_S2} ${TPU} ${OUTPUT}.scores_input ${OUTPUT}.scores_output_corr_tags
paste ${OUTPUT}.scores_input ${OUTPUT}.scores_output ${OUTPUT}.scores_output_corr_tags.scores > ${OUTPUT}.carb
