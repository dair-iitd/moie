# declare -a sizes=("small" "base" "large" "xl")
declare -a sizes=("base")
# declare -a splits=("dev" "test")
declare -a splits=("valid")
# declare -a models=("translation_combined_fa_en" "translation_combined_en_fa" "arabic_english_opus100")
declare -a models=("pt_mt5_vanilla")
for model in "${models[@]}"; do
  for size in "${sizes[@]}"; do
  for split in "${splits[@]}"; do
      echo "----------------"
      echo "* model: ${model}"
      echo "* size: ${size}"
      echo "* split: ${split}"
python evaluate_predictions.py \
  --eval_path=${model}/ \
  --eval_metric=translation --bucket_name=moie_bucket --dump



