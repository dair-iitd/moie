MODEL=$1
DEVICE_NAME=$2
INPUT=$3
OUTPUT=$4

TASK=$(basename $MODEL) 
CHECKPOINT_STEPS=1010000

if [[ $DEVICE_NAME == grpc* ]] # Run on TPU
then
  DATA_DIR="data/" python -m t5.models.mesh_transformer_main \
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
      --tpu="${DEVICE_NAME}"

  mv ${OUTPUT}-${CHECKPOINT_STEPS} ${OUTPUT}
else # Run on GPU
  DATA_DIR="data/" CUDA_VISIBLE_DEVICES="${DEVICE_NAME}" python -m t5.models.mesh_transformer_main \
      --module_import="t5_tasks" \
      --model_dir="${MODEL}" \
      --gin_file="dataset.gin" \
      --gin_file="${MODEL}/operative_config.gin" \
      --gin_file="infer.gin" \
      --gin_file="beam_search.gin" \
      --gin_param="input_filename = '${INPUT}'"\
      --gin_param="output_filename = '${OUTPUT}'" \
      --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
      --gin_param="utils.run.mesh_devices = ['gpu:0']" \
      --gin_param="MIXTURE_NAME = '${TASK}'" \
      --gin_param="utils.run.sequence_length = {'inputs': 256, 'targets': 128}" \
      --gin_param="utils.run.dataset_split = 'test'" \
      --gin_param="utils.run.batch_size=('tokens_per_batch', 4096)" \
      --gin_param="infer_checkpoint_step=${CHECKPOINT_STEPS}"

  mv ${OUTPUT}-${CHECKPOINT_STEPS} ${OUTPUT}
  python decode_file.py ${OUTPUT}
  # awk '{print substr($0,3)}' ${OUTPUT} > tmp && mv tmp ${OUTPUT}
  # awk '{print substr($0, 1, length($0)-1)}' ${OUTPUT} > tmp && mv tmp ${OUTPUT}
fi
