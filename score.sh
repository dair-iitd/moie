# export PROJECT=moie-313415
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

INPUT=$4
OUTPUT=$5

#  --gcp_project="${PROJECT}" \
#  --tpu_zone="${ZONE}" \

python -m t5.models.mesh_transformer_main \
     --module_import="t5_tasks" \
     --tpu="${TPU_NAME}" \
     --model_dir="${MODEL_DIR}" \
     --gin_file="dataset.gin" \
     --gin_file="${MODEL_DIR}/operative_config.gin" \
     --gin_file="score_from_file.gin" \
     --gin_file="beam_search.gin" \
     --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
     --gin_param="MIXTURE_NAME = '${TASK}'" \
     --gin_param="inputs_filename = '${BUCKET}/data/${INPUT}'"\
     --gin_param="targets_filename = '${BUCKET}/data/${OUTPUT}'" \
     --gin_param="scores_filename = '${BUCKET}/data/${OUTPUT}'" \
     --gin_param="utils.run.sequence_length = {'inputs': 256, 'targets': 128}" \
     --gin_param="utils.run.dataset_split = 'test'" \
     --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
     --gin_param="utils.run.eval_checkpoint_step=${FINETUNE_STEPS}" \
     --t5_tfds_data_dir="${BUCKET}/t5-tfds"


