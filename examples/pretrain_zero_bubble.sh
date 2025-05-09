#!/bin/bash


#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b

export CUDA_DEVICE_MAX_CONNECTIONS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export DTR_ENABLE=0
export CUDA_VISIBLE_DEVICES=1,2,3,4

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

DATASET_DIR='/tmp/zb_sample_dataset'
DATASET="${DATASET_DIR}/dataset/c4_text_document"
TOKENIZER="${DATASET_DIR}/tokenizers/tokenizer.model"


if [ ! -e "$DATASET"".idx" ]; then
  wget https://huggingface.co/datasets/ufotalent/zero_bubble_sample_dataset/resolve/main/zb_sample_dataset.tar.gz
  tar -xvf zb_sample_dataset.tar.gz -C /tmp
fi

# Running locally
if [ -z "$WORLD_SIZE" ]; then
  export WORLD_SIZE=1
  export RANK=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=22233
fi


GPUS_PER_NODE=2


if [ -z "$GPUS_PER_NODE" ]; then
  GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
  GPUS_PER_NODE=1
fi

if [ -z "$EXIT_INTERVAL" ]; then
  EXIT_INTERVAL=1000
fi

if [ -z "$LOG_INTERVAL" ]; then
  LOG_INTERVAL=1
fi

WORLD_SIZE_IN_GPUS=$(( $WORLD_SIZE * $GPUS_PER_NODE ))



if [ -z "$PIPELINE_SIZE" ]; then
  PIPELINE_SIZE=2
  LAYERS=8
  MICRO_BATCH_SIZE=1
  GLOBAL_BATCH_SIZE=8
  HIDDEN_SIZE=1024
  ATTENTION_HEADS=16
  ZERO_BUBBLE_MEM_LIMIT=$((1 * $PIPELINE_SIZE))
fi

profile_ranks="0 1"
for ((i = 1; i < $WORLD_SIZE_IN_GPUS; i++)); do
    profile_ranks="$profile_ranks $i"
done
if [ -z "$ZERO_BUBBLE_TIMER_START" ]; then
  ZERO_BUBBLE_TIMER_START=5
  ZERO_BUBBLE_TIMER_END=10
fi

if [ -z "$EVAL_INTERVAL" ]; then
  EVAL_INTERVAL=10000
fi

if [ -z "$TP_SIZE" ]; then
  TP_SIZE=2
fi



options=" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 4 \
  --num-layers 12 \
  --hidden-size 768 \
  --num-attention-heads 16 \
  --exit-interval $EXIT_INTERVAL \
  --seq-length 512\
  --max-position-embeddings 1024 \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --group-query-attention \
  --num-query-groups 1 \
  --train-samples 160 \
  --lr-decay-samples 2 \
  --lr-warmup-samples 1 \
  --lr 6.0e-5 \
  --min-lr 6.0e-6 \
  --lr-decay-style cosine \
  --log-interval ${LOG_INTERVAL} \
  --eval-iters 0 \
  --eval-interval $EVAL_INTERVAL \
  --data-path ${DATASET} \
  --tokenizer-type GPTSentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER} \
  --split 98,2,0 \
  --clip-grad 8.0 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.006 \
  --no-barrier-with-level-1-timing \
  --profile-step-start 2 \
  --profile-step-end 5 \
  --untie-embeddings-and-output-weights \
  --use-legacy-models \
  
  --use-flash-attn \
  --transformer-impl local \
  --use-distributed-optimizer \
  --enable-zb-runtime \
  --no-create-attention-mask-in-dataloader \
  --profile-ranks $profile_ranks 
  
  "

if [ -z "$FP32" ]; then
  options="$options --fp16"
fi

if [ ! -z "$PROFILED" ]; then
  options="$options --profile"
fi

ZERO_BUBBLE_V_SCHEDULE=1

if [ ! -z "$ZERO_BUBBLE_V_SCHEDULE" ]; then
  ENABLE_ZERO_BUBBLE=1
  options="$options --zero-bubble-v-schedule "
fi

if [ ! -z "$ENABLE_ZERO_BUBBLE" ]; then
  options="$options --enable-zero-bubble \
  --zero-bubble-pipeline-timers-start-iter $ZERO_BUBBLE_TIMER_START \
  --zero-bubble-pipeline-timers-end-iter $ZERO_BUBBLE_TIMER_END \
  --zero-bubble-max-pending-backward $ZERO_BUBBLE_MEM_LIMIT"
  if [ -z "$FP32" ]; then
    if [ -z "$SYNC_OPTIMIZER" ]; then
        options="$options --enable-optimizer-post-validation"
    fi
  fi
fi

if [ ! -z "$ENABLE_EXACTLY_NUMERIC_MATCH" ]; then
  options="$options --enable-exactly-numeric-match \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0"
fi

if [ ! -z "$INTERLEAVED_1F1B" ]; then
  options="$options --num-layers-per-virtual-pipeline-stage 1"
fi

run_cmd="torchrun --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --nproc_per_node=$GPUS_PER_NODE ${DIR}/pretrain_gpt.py $@ ${options} ${EXTRA_OPTIONS}\
  --distributed-backend  nccl
  " \
  
PROFILED=1
AIP_RUN_NAME=test

if [ ! -z "$PROFILED" ]; then
  run_cmd="nsys profile -s none -t nvtx,cuda \
    --output $AIP_RUN_NAME.$RANK.nsys-rep \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $run_cmd"
fi

echo $CUDA_VISIBLE_DEVICES
echo $run_cmd
# sleep 100000
eval $run_cmd
#eval $run_cmd > >(tee log.$AIP_RUN_NAME) 2>&1

set +x
