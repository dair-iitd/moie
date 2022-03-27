MODEL=$1
TPU_NAME=$2
INPUT=$3
OUTPUT=$4

# export BUCKET=gs://moie_bucket
#  --gcp_project="${PROJECT}" \
#  --tpu_zone="${ZONE}" \
#  --t5_tfds_data_dir="${BUCKET}/t5-tfds"

TASK=$(basename $MODEL) 
# MODEL_DIR="${BUCKET}/models/${DIR}"
CHECKPOINT_STEPS=1010000

CUDA_VISIBLE_DEVICES=0 python -m t5.models.mesh_transformer_main \
     --module_import="t5_tasks" \
     --model_dir="${MODEL}" \
     --gin_file="dataset.gin" \
     --gin_file="${MODEL}/operative_config.gin" \
     --gin_file="infer.gin" \
     --gin_file="beam_search.gin" \
     --gin_param="input_filename = '${INPUT}'"\
     --gin_param="output_filename = '${OUTPUT}'" \
     --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
     --gin_param="MIXTURE_NAME = '${TASK}'" \
     --gin_param="utils.run.sequence_length = {'inputs': 256, 'targets': 128}" \
     --gin_param="utils.run.dataset_split = 'test'" \
     --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
     --gin_param="infer_checkpoint_step=${CHECKPOINT_STEPS}" \
     --tpu="${TPU_NAME}"

mv ${OUTPUT}-${CHECKPOINT_STEPS} ${OUTPUT}
