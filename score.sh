MODEL=$1
TPU_NAME=$2
INPUT=$3
OUTPUT=$4

TASK=$(basename $MODEL)
FINETUNE_STEPS=10000

python -m t5.models.mesh_transformer_main \
     --module_import="t5_tasks" \
     --tpu="${TPU_NAME}" \
     --model_dir="${MODEL}" \
     --gin_file="dataset.gin" \
     --gin_file="${MODEL}/operative_config.gin" \
     --gin_file="score_from_file.gin" \
     --gin_file="beam_search.gin" \
     --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-8'" \
     --gin_param="MIXTURE_NAME = '${TASK}'" \
     --gin_param="inputs_filename = '${INPUT}'"\
     --gin_param="targets_filename = '${OUTPUT}'" \
     --gin_param="scores_filename = '${OUTPUT}'" \
     --gin_param="utils.run.sequence_length = {'inputs': 256, 'targets': 128}" \
     --gin_param="utils.run.dataset_split = 'test'" \
     --gin_param="utils.run.batch_size=('tokens_per_batch', 24576)" \
     --gin_param="utils.run.eval_checkpoint_step=${FINETUNE_STEPS}" 
