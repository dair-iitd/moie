export PROJECT=moie-313415
export ZONE=europe-west4-a
export TPU_NAME=$2
export BUCKET=gs://moie_bucket

SIZE="base"
TASK=$1
PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
MODEL_DIR="${BUCKET}/models/${TASK}"

PRETRAINED_STEPS=1000000
# FINETUNE_STEPS=20000
FINETUNE_STEPS=$3

# Run fine-tuning
python -m t5.models.mesh_transformer_main \
      --module_import="t5_tasks" \
      --tpu="${TPU_NAME}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --model_dir="${MODEL_DIR}" \
      --gin_file="dataset.gin" \
      --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
      --gin_param="utils.run.save_checkpoints_steps=2000" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.sequence_length = {'inputs': 256, 'targets': 128}" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
      --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
      --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
      --t5_tfds_data_dir="${BUCKET}/t5-tfds" \
      --gin_location_prefix="multilingual_t5/gin/"

python clean_ckpt.py $HOME/moie_bucket/models/${TASK}

# python -m t5.models.mesh_transformer_main \
#      --module_import="t5_tasks" \
#      --tpu="${TPU_NAME}" \
#      --gcp_project="${PROJECT}" \
#      --tpu_zone="${ZONE}" \
#      --model_dir="${MODEL_DIR}" \
#      --gin_file="dataset.gin" \
#      --gin_file="${MODEL_DIR}/operative_config.gin" \
#      --gin_file="eval.gin" \
#      --gin_file="beam_search.gin" \
#      --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
#      --gin_param="MIXTURE_NAME = '${TASK}'" \
#      --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 512}" \
#      --gin_param="utils.run.dataset_split = 'valid'" \
#      --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
#      --gin_param="utils.run.eval_checkpoint_step='all'" \
#      --t5_tfds_data_dir="${BUCKET}/t5-tfds"

# python evaluate_predictions.py \
#   --eval_path=models/${TASK}/valid_eval \
#   --eval_metric=translation --bucket_name=moie_bucket --dump

