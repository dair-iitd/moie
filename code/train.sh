TASK=$1
MODEL_DIR=$2
DATA_DIR=$3
DEVICE_NAME=$4
FINETUNE_STEPS=$5

PRETRAINED_DIR="gs://t5-data/pretrained_models/mt5/base"
PRETRAINED_STEPS=1000000

if [[ $DEVICE_NAME == gprc* ]] # Can evaluate gen2oie scoring only on TPU
then
  ## need to set project and zone if not running in colab
  #export PROJECT=xxx 
  #export ZONE=yyy
  BUCKET=gs://moie_bucket
  DATA_DIR="${DATA_DIR}/" python -m t5.models.mesh_transformer_main \
        --module_import="t5_tasks" \
        --tpu="${DEVICE_NAME}" \
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
        --gin_location_prefix="multilingual_t5/gin/"
else
  DATA_DIR="${DATA_DIR}/" CUDA_VISIBLE_DEVICES=${DEVICE_NAME} python -m t5.models.mesh_transformer_main \
        --module_import="t5_tasks" \
        --model_dir="${MODEL_DIR}" \
        --gin_file="dataset.gin" \
        --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
        --gin_param="utils.run.save_checkpoints_steps=2000" \
        --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
        --gin_param="utils.run.mesh_devices = ['gpu:0']" \
        --gin_param="MIXTURE_NAME = '${TASK}'" \
        --gin_param="utils.run.sequence_length = {'inputs': 128, 'targets': 128}" \
        --gin_param="utils.run.batch_size=('tokens_per_batch', 1024)" \
        --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS + FINETUNE_STEPS))" \
        --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
        --gin_location_prefix="multilingual_t5/gin/"
fi