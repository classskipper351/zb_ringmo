
#export CUDA_DEVICE_MAX_CONNECTIONS=1


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # necessary for multi node


export SNAP_FILE_NAME="pretrain_ringmo_p2_5iter"
export DTR_ENABLE=0
export MEM_BUDGET=8      # only budget > 0 can use RESIDUAL_DEGREE, otherwise reserve leak
export RECORD_MEM_SNAPSHOT=0
export RESIDUAL_DEGREE=6
export COST_FIRST_EVICT=0
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
DATA_PATH_TRAIN=/data/yaochenguang/imagenet-1k/imagnet256k.json
DATA_PATH_VAL=/data/yaochenguang/imagenet-1k/imagnet256k.json
CHECKPOINT_PATH=/home/yaochenguang/data/Megatron-LM/checkpoints

RINGMO_ARGS="
        --num-layers-per-virtual-pipeline-stage 2 \
        --vision-pretraining-type dino \
   	    --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 2\
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --patch-dim 4 \
        --seq-length 3136 \
        --max-position-embeddings 3136 \
        --img-h 192 \
        --img-w 192 \
        --mask-factor 1.0 \
        --fp16 \
        --train-iters 10 \
        --lr-decay-style cosine \
        --micro-batch-size 4 \
        --global-batch-size 128 \
        --lr 0.0005 \
        --min-lr 0.00001 \
        --attention-dropout 0.0 \
        --weight-decay 0.05 \
        --lr-warmup-iters 1\
        --clip-grad 1.0 \
        --no-gradient-accumulation-fusion \
        --num-workers 1 \
        --no-async-tensor-model-parallel-allreduce
        "

GPUS_PER_NODE=
MASTER_ADDR=localhost
MASTER_PORT=22233
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"



DATA_ARGS="
     --dataloader-type external
     --tokenizer-type NullTokenizer \
     --vocab-size 0 \
     --data-path  $DATA_PATH_TRAIN $DATA_PATH_VAL\
     --no-data-sharding \
"

TENSOR_BOARD_DIR=/home/yaochenguang/data/Megatron-LM/logs

OUTPUT_ARGS="
     --log-interval 5 \
     --save-interval 1000000000 \
     --eval-interval 1 \
     --eval-iters 1\
     --tensorboard-dir ${TENSOR_BOARD_DIR} \
     
"

PROFILER_ARGS="
     
     --profile-step-start 1\
     --profile-step-end 20\
     --profile-memory \
     --profile-stack \
     --profile-ranks 0 1 \
     --profile-interval 1\

"


OPEN_DEBUG=1

if [ -d "$TENSOR_BOARD_DIR" ]; then
    echo "Cleaning TensorBoard directory: $TENSOR_BOARD_DIR"
    rm -rf "${TENSOR_BOARD_DIR}"/*
else
    echo "Creating TensorBoard directory: $TENSOR_BOARD_DIR"
    mkdir -p "$TENSOR_BOARD_DIR"
fi

torchrun $DISTRIBUTED_ARGS  swin_pretrain_pp.py \
    $RINGMO_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $PROFILER_ARGS\
    --open-debug $OPEN_DEBUG \
    --distributed-backend nccl\
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH
     