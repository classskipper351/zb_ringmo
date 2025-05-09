#!/bin/bash

# 用户选择的方法（从 run_selected_schedule.sh 继承）
METHOD=zb

# 检查方法是否有效
VALID_METHODS=("1f1b" "1f1bv" "offload-grouped-interleaved" "zb" "zbv" "zbv-half" "zbv-min" "seq1f1b")
if [[ ! " ${VALID_METHODS[@]} " =~ " ${METHOD} " ]]; then
    echo "无效的方法: $METHOD。请从以下选项中选择: ${VALID_METHODS[@]}"
    exit 1
fi

# 环境设置（部分从 pretrain_swin_distributed.sh 继承）
export CUDA_DEVICE_MAX_CONNECTIONS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export DTR_ENABLE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 数据集路径（从 pretrain_swin_distributed.sh 继承）
DATA_PATH_TRAIN=/data/yaochenguang/imagenet-1k/imagnet256k.json
DATA_PATH_VAL=/data/yaochenguang/imagenet-1k/imagnet256k.json

# 本地运行时的默认设置（从 run_selected_schedule.sh 继承）
if [ -z "$WORLD_SIZE" ]; then
    export WORLD_SIZE=1
    export RANK=0
    export MASTER_ADDR=localhost
    export MASTER_PORT=22233
fi

# GPU 设置（从 pretrain_swin_distributed.sh 继承）
GPUS_PER_NODE=4
WORLD_SIZE_IN_GPUS=$(( $WORLD_SIZE * $GPUS_PER_NODE ))

# 训练参数（从 pretrain_swin_distributed.sh 继承，并参考 run_selected_schedule.sh 的结构）
EXIT_INTERVAL=1000000000
LOG_INTERVAL=2
PIPELINE_SIZE=4
LAYERS=12
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32
HIDDEN_SIZE=768
ATTENTION_HEADS=12
ZERO_BUBBLE_MEM_LIMIT=$((1 * $PIPELINE_SIZE))
ZERO_BUBBLE_TIMER_START=10
ZERO_BUBBLE_TIMER_END=20
EVAL_INTERVAL=1
EVAL_ITERS=1
TP_SIZE=1
#export PROFILED=0
AIP_RUN_NAME="pretrain_swin_${METHOD}"
export MEM_BUDGET=8
export RESIDUAL_DEGREE=6
export COST_FIRST_EVICT=0

# 日志目录（从 run_selected_schedule.sh 继承）
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
TENSORBOARD_DIR=$DIR/logs/tensorboard
mkdir -p $TENSORBOARD_DIR
S
# 清理或创建 TensorBoard 目录（从 pretrain_swin_distributed.sh 继承）
if [ -d "$TENSORBOARD_DIR" ]; then
    echo "Cleaning TensorBoard directory: $TENSORBOARD_DIR"
    rm -rf "${TENSORBOARD_DIR}"/*
else
    echo "Creating TensorBoard directory: $TENSORBOARD_DIR"
    mkdir -p "$TENSORBOARD_DIR"
fi

# 根据方法设置调度选项（从 run_selected_schedule.sh 继承）
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
# 构建训练参数（基于 pretrain_swin_distributed.sh，适配 Zero Bubble 框架）
options=" \
  --vision-pretraining-type dino \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PIPELINE_SIZE \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $ATTENTION_HEADS \
  --patch-dim 4 \
  --seq-length 3136 \
  --max-position-embeddings 3136 \
  --img-h 192 \
  --img-w 192 \
  --mask-factor 0.6 \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --train-iters 60\
  --lr 0.0005 \
  --min-lr 0.00001 \
  --lr-decay-style cosine \
  --lr-warmup-iters 1 \
  --clip-grad 1.0 \
  --weight-decay 0.05 \
  --attention-dropout 0.0 \
  --log-interval $LOG_INTERVAL \
  --eval-iters $EVAL_ITERS \
  --eval-interval $EVAL_INTERVAL \
  --data-path $DATA_PATH_TRAIN $DATA_PATH_VAL \
  --dataloader-type external \
  --tokenizer-type NullTokenizer \
  --vocab-size 0 \
  --no-data-sharding \
  --no-gradient-accumulation-fusion \
  --num-workers 0 \
  --no-async-tensor-model-parallel-allreduce \
  --tensorboard-dir $TENSORBOARD_DIR \
  --profile\
  --profile-step-start 10 \
  --profile-step-end 40 \
  --profile-ranks 0 1\
  --open-debug 1 \
  --enable-zb-runtime \
  --uniformed-compute-time $UNIFORM\
  --transformer-impl local\
  "
# 添加 fp16 选项
options="$options --fp16"

# 根据调度选项动态添加参数（从 run_selected_schedule.sh 继承）
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
  --nproc_per_node=$GPUS_PER_NODE ${DIR}/swin_pretrain_pp.py $options $EXTRA_OPTIONS \
  --distributed-backend nccl"

# 如果启用了 profiling，则添加 nsys profile（从 run_selected_schedule.sh 继承）
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
    --output ringmo$AIP_RUN_NAME.$RANK.$UNIFORM_FLAG.$TIMESTAMP.nsys-rep \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $run_cmd"
fi

# 输出并执行命令
echo "使用方法 $METHOD 运行 Swin 模型训练..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "运行命令: $run_cmd"
eval $run_cmd