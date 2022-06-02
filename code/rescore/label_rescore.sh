MODEL_DIR=$1
FILE=$2

python allennlp2extlbl.py --inp ${FILE} --out ${FILE}.ext_sent_lbls
python run.py --save ${MODEL_DIR} --mode predict --model_str bert-base-multilingual-cased --task oie --gpus 1 --inp ${FILE}.ext_sent_lbls --out ${FILE}.ext_sent_lbls.preds
python allennlp2extlbl.py --inp ${FILE} --out ${FILE}.ext_sent_lbls --openie6_predictions ${FILE}.ext_sent_lbls.preds
python utils/oie_to_allennlp.py --inp ${FILE}.ext_sent_lbls.preds2.oie --out ${FILE}.carb
