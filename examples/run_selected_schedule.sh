#!/bin/bash

# 用户选择的方法
METHOD=zb

# 检查方法是否有效
VALID_METHODS=("1f1b" "1f1bv" "offload-grouped-interleaved" "zb" "zbv" "zbv-half" "zbv-min" "seq1f1b")
if [[ ! " ${VALID_METHODS[@]} " =~ " ${METHOD} " ]]; then
    echo "无效的方法: $METHOD。请从以下选项中选择: ${VALID_METHODS[@]}"
    exit 1
fi

# 从 pretrain_zero_bubble.sh 中继承的环境设置
export CUDA_DEVICE_MAX_CONNECTIONS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export DTR_ENABLE=0
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 数据集和 tokenizer 路径
DATASET_DIR='/tmp/zb_sample_dataset'
DATASET="${DATASET_DIR}/dataset/c4_text_document"
TOKENIZER="${DATASET_DIR}/tokenizers/tokenizer.model"

# 如果数据集不存在，下载并解压
if [ ! -e "$DATASET.idx" ]; then
    wget https://huggingface.co/datasets/ufotalent/zero_bubble_sample_dataset/resolve/main/zb_sample_dataset.tar.gz
    tar -xvf zb_sample_dataset.tar.gz -C /tmp
fi

# 本地运行时的默认设置
if [ -z "$WORLD_SIZE" ]; then
    export WORLD_SIZE=1
    export RANK=0
    export MASTER_ADDR=localhost
    export MASTER_PORT=22233
fi

# GPU 
GPUS_PER_NODE=2
WORLD_SIZE_IN_GPUS=$(( $WORLD_SIZE * $GPUS_PER_NODE ))

# 从 pretrain_zero_bubble.sh 中继承的训练参数
EXIT_INTERVAL=1000
LOG_INTERVAL=5
PIPELINE_SIZE=2
LAYERS=12
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
HIDDEN_SIZE=768
ATTENTION_HEADS=16
ZERO_BUBBLE_MEM_LIMIT=$((1 * $PIPELINE_SIZE))
ZERO_BUBBLE_TIMER_START=10
ZERO_BUBBLE_TIMER_END=20
EVAL_INTERVAL=10000
TP_SIZE=1
#export  PROFILED=0
AIP_RUN_NAME="test_${METHOD}"

# 日志目录
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

# 根据方法设置调度选项（参考 test_all_schedules.sh）
export ENABLE_ZERO_BUBBLE=
export INTERLEAVED_1F1B=
export ZERO_BUBBLE_V_SCHEDULE=
export EXTRA_OPTIONS='--allow-padding-num-layers'

case $METHOD in
    "1f1b")
        export INTERLEAVED_1F1B=
        ;;
    "1f1bv")
        export INTERLEAVED_1F1B=
        export EXTRA_OPTIONS="${EXTRA_OPTIONS} --enable-1f1b-v"
        ;;
    "zb")
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
        ;;
    "zbv")
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
        export ZERO_BUBBLE_V_SCHEDULE=1
        ;;
    "zbv-half")
        export SYNC_OPTIMIZER=1
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
        export ZERO_BUBBLE_V_SCHEDULE=1
        export EXTRA_OPTIONS="${EXTRA_OPTIONS} --zero-bubble-v-schedule-mem-setup half"
        ;;
    "zbv-min")
        export SYNC_OPTIMIZER=1
        export ENABLE_ZERO_BUBBLE=1
        export ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
        export ZERO_BUBBLE_V_SCHEDULE=1
        export EXTRA_OPTIONS="${EXTRA_OPTIONS} --zero-bubble-v-schedule-mem-setup min"
        ;;
    "seq1f1b")
        export EXTRA_OPTIONS="${EXTRA_OPTIONS} --num-seq-splits 2"
        ;;
    "offload-grouped-interleaved")
        export INTERLEAVED_1F1B=1
        export INTERLEAVE_GROUP=2
        export OFFLOAD_TIME='0.2'
        export OFFLOAD_CHUNK_NUM=4
        export OFFLOAD=1
        ;;
    *)
        echo "未知方法: $METHOD"
        exit 1
        ;;
esac
UNIFORM=0
# 构建训练参数（从 pretrain_zero_bubble.sh 中继承）
options=" \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PIPELINE_SIZE \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $ATTENTION_HEADS \
  --exit-interval $EXIT_INTERVAL \
  --seq-length 512 \
  --max-position-embeddings 1024 \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --group-query-attention \
  --num-query-groups 1 \
  --train-samples 800 \
  --lr-decay-samples 2 \
  --lr-warmup-samples 1 \
  --lr 6.0e-5 \
  --min-lr 6.0e-6 \
  --lr-decay-style cosine \
  --log-interval $LOG_INTERVAL \
  --eval-iters 0 \
  --eval-interval $EVAL_INTERVAL \
  --data-path $DATASET \
  --tokenizer-type GPTSentencePieceTokenizer \
  --tokenizer-model $TOKENIZER \
  --split 98,2,0 \
  --clip-grad 8.0 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.006 \
  --no-barrier-with-level-1-timing \
  --profile\
  --profile-step-start 10 \
  --profile-step-end 40 \
  --untie-embeddings-and-output-weights \
  --use-legacy-models \
  --use-flash-attn \
  --transformer-impl local \
  --use-distributed-optimizer \
  --enable-zb-runtime \
  --no-create-attention-mask-in-dataloader \
  --profile-ranks 0 1 2 3\
  --open-debug 1 \
  --uniformed-compute-time $UNIFORM\
  "

# 添加 fp16 选项
options="$options --fp16"

# 根据调度选项动态添加参数
if [ ! -z "$ENABLE_ZERO_BUBBLE" ]; then
    options="$options --enable-zero-bubble \
    --zero-bubble-pipeline-timers-start-iter $ZERO_BUBBLE_TIMER_START \
    --zero-bubble-pipeline-timers-end-iter $ZERO_BUBBLE_TIMER_END \
    --zero-bubble-max-pending-backward $ZERO_BUBBLE_MEM_LIMIT"
fi

if [ ! -z "$ZERO_BUBBLE_V_SCHEDULE" ]; then
    options="$options --zero-bubble-v-schedule"
fi

if [ ! -z "$INTERLEAVED_1F1B" ]; then
    options="$options --num-layers-per-virtual-pipeline-stage 1"
fi

# 构建运行命令
run_cmd="torchrun --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --nproc_per_node=$GPUS_PER_NODE ${DIR}/pretrain_gpt.py $options $EXTRA_OPTIONS \
  --distributed-backend nccl"

# 如果启用了 profiling，则添加 nsys profile
if [ ! -z "$PROFILED" ]; then
    # 获取当前时间戳（格式：YYYYMMDD_HHMMSS）
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    
    # 根据UNIFORM参数添加标识
    if [ "$UNIFORM" -eq 1 ]; then
        UNIFORM_FLAG="uniformed"
    else
        UNIFORM_FLAG="nonuniformed"
    fi
    
    run_cmd="nsys profile -s none -t nvtx,cuda \
    --output gptpp$PIPELINE_SIZE_$AIP_RUN_NAME.$RANK.$UNIFORM_FLAG.$TIMESTAMP.nsys-rep \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $run_cmd"
fi

# 输出并执行命令
echo "使用方法 $METHOD 运行训练..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "运行命令: $run_cmd"
eval $run_cmd