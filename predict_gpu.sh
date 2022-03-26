# export PROJECT=moie-313415
# export ZONE=europe-west4-a
# export TPU_NAME=$2
export BUCKET=gs://moie_bucket

SIZE="base"
DIR=$1
TASK=${DIR//\//_}
# TASK='pt_gen2oie_s2_ctranslate_clp'
PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/${SIZE}"
# MODEL_DIR="../../../models/pt_gen2oie_s2_ctranslate_clp"
MODEL_DIR="${BUCKET}/models/${DIR}"
DATA_DIR="${BUCKET}/data/${DIR}"

PRETRAINED_STEPS=1000000
# CHECKPOINT_STEPS=1010000
CHECKPOINT_STEPS=$2

# INPUT='../../../code/carb/data/pt_test.input'
# OUTPUT='pt_test.output.temp'
INPUT=$3
OUTPUT=$4

#--tpu="${TPU_NAME}" \
#--gcp_project="${PROJECT}" \
#--tpu_zone="${ZONE}" \
#--gin_param="input_filename = '${BUCKET}/data/${INPUT}'"\
#--gin_param="output_filename = '${BUCKET}/data/${OUTPUT}'" \

CUDA_VISIBLE_DEVICES= python -m t5.models.mesh_transformer_main \
     --module_import="t5_tasks" \
     --model_dir="${MODEL_DIR}" \
     --gin_file="dataset.gin" \
     --gin_file="${MODEL_DIR}/operative_config.gin" \
     --gin_file="infer.gin" \
     --gin_file="beam_search.gin" \
     --gin_param="input_filename = '${INPUT}'"\
     --gin_param="output_filename = '${OUTPUT}'" \
     --gin_param="MIXTURE_NAME = '${TASK}'" \
     --gin_param="utils.run.sequence_length = {'inputs': 256, 'targets': 128}" \
     --gin_param="utils.run.dataset_split = 'test'" \
     --gin_param="utils.run.batch_size=('tokens_per_batch', 4096)" \
     --gin_param="infer_checkpoint_step=${CHECKPOINT_STEPS}" \
     --t5_tfds_data_dir="${BUCKET}/t5-tfds"

# gsutil mv ${BUCKET}/data/${OUTPUT}-${CHECKPOINT_STEPS} ${BUCKET}/data/${OUTPUT}
