DATA_TYPE=ctranslate_clp
LANGUAGE=es
mkdir -p ../data/${LANGUAGE}/${DATA_TYPE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python aligner.py --inp1 ../data/openie6/train.extractions_labels  --inp2 ../data/${LANGUAGE}/mbart/consistent/train.extractions  --output_file ../data/${LANGUAGE}/${DATA_TYPE}/aligned.extractions --alignment_type clp_sentence  --lang ${LANGUAGE} --model_name_or_path ../models/${LANGUAGE}/model_without_co/ 
python generative_data.py --fp1 ../data/${LANGUAGE}/${DATA_TYPE}/aligned.extractions  --fp2 ../data/openie6/train.count_extractions --out ../data/${LANGUAGE}/${DATA_TYPE}/train.target
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_valid_split.py --model_type genoie --data_type ${DATA_TYPE} --lang ${LANGUAGE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_valid_split.py --model_type gen2oie_s1 --data_type ${DATA_TYPE} --lang ${LANGUAGE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python train_valid_split.py --model_type gen2oie_s2 --data_type ${DATA_TYPE} --lang ${LANGUAGE}
bash preprocess_train.sh ${LANGUAGE} genoie ${DATA_TYPE}
bash preprocess_train.sh ${LANGUAGE} gen2oie_s1 ${DATA_TYPE}
bash preprocess_train.sh ${LANGUAGE} gen2oie_s2 ${DATA_TYPE}

