set -ex
ENV=${ENV:-dsw}
### BASE CONFIG ###
DEFAULT_MODEL_SIZE=16B
MODEL_SIZE=${MODEL_SIZE:-${DEFAULT_MODEL_SIZE}}
BATCH_SIZE=${BATCH_SIZE:-5}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4800}
DEFAULT_LR=4.2E-4
LR=${LR:-${DEFAULT_LR}}
DEFAULT_MIN_LR=4.2E-5
MIN_LR=${MIN_LR:-${DEFAULT_MIN_LR}}
INIT_METHOD_STD=${INIT_METHOD_STD:-0.006} # 0.006 

SEQ_LEN=${SEQ_LEN:-4096}
PAD_LEN=100
PR=${PR:-bf16}
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
PP=${PP:-1} 
EP=${EP:-4}
FL=${FLASH_ATTENTION:-true} # true
TP=1
CP=1
SP=false
DO=true
SFT=false
### PARALLEL / BOOL OPTION ###

### OTHERS ###
AC=${AC:-full} # full
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
if [[ -z $DATASET_FILE ]] ; then
    echo "Missing environment variable DATASET_FILE."
    exit 1
fi
DATASET_PATH="$(cat ${DATASET_FILE})"
VALID_DATASET_PATH=${DATASET_PATH}
OUTPUT_DIR=${OUTPUT_DIR}

if [[ -z $TOKENIZER_PATH ]] ; then
    echo "Missing environment variable TOKENIZER_PATH."
    exit 1
fi

CKPT_FORMAT=${CKPT_FORMAT:-torch_dist}
if [ ${CKPT_FORMAT} = torch_dist_async ] ; then
    ckpt_options=" --ckpt-format torch_dist --async-save "
elif [ ${CKPT_FORMAT} = torch_dist_no_optim ] ; then
    ckpt_options=" --ckpt-format torch_dist --no-save-optim "
elif [ ${CKPT_FORMAT} = torch_dist ] ; then
    ckpt_options=" --ckpt-format torch_dist "
elif [ ${CKPT_FORMAT} = torch ] ; then
    ckpt_options=" --ckpt-format torch "
fi

# DEBUG model without saving optim
if [[ ${DEBUG_PRETRAIN_CHECKPOINT_PATH:-none} != none ]]; then
    PRETRAIN_CHECKPOINT_PATH=$DEBUG_PRETRAIN_CHECKPOINT_PATH
    ckpt_options=" ${ckpt_options} \
        --auto-detect-ckpt-format \
        --no-load-optim \
        --no-load-rng \
        --no-save-optim \
        --no-save-rng \
        "
fi

# training configuraitons
TRAIN_TOKENS=${TRAIN_TOKENS:-7500000000000}
WARMUP_TOKENS=${WARMUP_TOKENS:-37748736000}

OUTPUT_BASEPATH=${OUTPUT_DIR}
### OTHERS ###
if [[ ${DEBUG} = on ]] ; then
    export NVTE_DEBUG=1
    export NVTE_DEBUG_LEVEL=2
    export CUDNN_LOGERR_DBG=1
    export CUDNN_LOGDEST_DBG=stderr
fi

if [[ ${CHECK_NAN:-true} = false ]]; then
    new_options=" ${new_options} --no-check-for-nan-in-loss-and-grad"
fi

### Begin of Script ###
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
if [ -z $MEGATRON_PATH ]; then
    MEGATRON_PATH=$( dirname ${CURRENT_DIR})
fi
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_BIAS_GELU_NVFUSION=0

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ $ENV = dsw ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=8
elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    fl_options=" --attention-backend flash "
elif [ $FL = false ]; then
    fl_options=" --attention-backend unfused "
fi

HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
NUM_LAYERS=${NUM_LAYERS:-28}
INTERMEDIATE_SIZE=10944
MOE_INTERMEDIATE_SIZE=1408
MAX_POSITION_EMBEDDINGS=163840 #${SEQ_LEN}
EXTRA_VOCAB_SIZE=256
KV_LORA_RANK=512
QK_NOPE_HEAD_DIM=${QK_NOPE_HEAD_DIM:-128}
QK_ROPE_HEAD_DIM=64
V_HEAD_DIM=128
ROPE_THETA=10000
SCALE_FACTOR=40
NUM_EXPERTS=${NUM_EXPERTS:-64}
ROUTER_TOPK=6
NUM_SHARED_EXPERTS=2
MOE_LAYER_FREQ=1
RMS_NORM_EPS=1e-6
MOE_FIRST_K_DENSE_REPLACE=0

moe_options=" \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --moe-router-topk ${ROUTER_TOPK} \
    --num-experts ${NUM_EXPERTS} \
    --moe-layer-freq ${MOE_LAYER_FREQ} \
    --moe-first-k-dense-replace ${MOE_FIRST_K_DENSE_REPLACE} \
    --moe-aux-loss-coeff 0.001 \
    --moe-shared-expert-intermediate-size $((${MOE_INTERMEDIATE_SIZE} * ${NUM_SHARED_EXPERTS} )) \
    --expert-model-parallel-size ${EP} \
    --kv-lora-rank ${KV_LORA_RANK} \
    --qk-head-dim ${QK_NOPE_HEAD_DIM} \
    --qk-pos-emb-head-dim ${QK_ROPE_HEAD_DIM} \
    --v-head-dim ${V_HEAD_DIM} \
    --moe-grouped-gemm \
    --moe-router-topk-scaling-factor 1.0"

LOAD_BALANCE_TYPE=${LOAD_BALANCE_TYPE:-aux_loss}
if [ ${LOAD_BALANCE_TYPE} = seq_aux_loss ] ; then
    moe_options=" ${moe_options} --moe-router-load-balancing-type ${LOAD_BALANCE_TYPE} "
elif [ ${LOAD_BALANCE_TYPE} = aux_loss ] ; then
    moe_options=" ${moe_options}  --moe-router-load-balancing-type ${LOAD_BALANCE_TYPE} "
else
echo "Unsupported moe-router-load-balancing-type: ${LOAD_BALANCE_TYPE}"
exit 1
fi

MOE_ROUTER_GROUPS=${MOE_ROUTER_GROUPS:-0} # 8
MOE_ROUTER_GROUPS_TOPK=${MOE_ROUTER_GROUPS_TOPK:-0} # 4

if [ $MOE_ROUTER_GROUPS -gt 0 ] && [ $MOE_ROUTER_GROUPS_TOPK -gt 0 ]; then
    moe_options=" ${moe_options}  --moe-router-num-groups ${MOE_ROUTER_GROUPS} \
    --moe-router-group-topk ${MOE_ROUTER_GROUPS_TOPK} "
fi

if [[ ${ROUTER_TOPK_SCALING_FACTOR:-none} != none ]]; then
moe_options=" ${moe_options} --moe-router-topk-scaling-factor ${ROUTER_TOPK_SCALING_FACTOR} "
fi

if [[ ${ROUTER_BIAS:-false} = true ]]; then
moe_options=" ${moe_options} --moe-router-enable-expert-bias \
            --moe-router-bias-update-rate 1e-3"
fi

DISPATCHER_TYPE=${DISPATCHER_TYPE:-flex_deepep}
if [ $DISPATCHER_TYPE = flex_deepep ]; then
    moe_options=" ${moe_options} --moe-token-dispatcher-type flex --moe-enable-deepep "
else
echo "Unsupported dispatcher type: ${DISPATCHER_TYPE}"
exit 1
fi

ROUTER_SCORE_FUNC=${ROUTER_SCORE_FUNC:-pre_softmax}
if [ $ROUTER_SCORE_FUNC = sigmod ]; then
    moe_options=" ${moe_options}  --moe-router-score-function sigmoid  "
elif [ $ROUTER_SCORE_FUNC = softmax ]; then
    moe_options=" ${moe_options}  --moe-router-score-function softmax "
elif [ $ROUTER_SCORE_FUNC = pre_softmax ]; then
    moe_options=" ${moe_options} --moe-router-score-function softmax --moe-router-pre-softmax "
else
echo "Unsupported router score function: ${ROUTER_SCORE_FUNC}"
exit 1
fi

# For MoE Stability
if [[ ${WARMUP_ROUTER:-0} -gt 0 ]]; then
    moe_options=" ${moe_options}  --moe-warmup-router  ${WARMUP_ROUTER}  "
fi

if [ ! -z ${APPLY_NORM_HEAD} ];then
    moe_options=" ${moe_options}  --moe-apply-norm-head "
fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"
 
if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS:-1} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-granularity selective \
        --recompute-modules ${RECOMPUTE_MODULES:-"core_attn moe_act layernorm mla_up_proj mlp moe"} \
    "
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = blockwise ]; then
    pr_options=" \
        --bf16 \
        --fp8-format e4m3  \
        --fp8-recipe blockwise \
    "
    export FP8_OUTER=${FP8_OUTER:-true}

    if [ ${FP8_DATAFLOW:-false} = true ]; then
        pr_options=" ${pr_options} --moe-fp8-flow"

        if [ ${FUSED_PERM_PAD:-true} = true ]; then
            pr_options=" ${pr_options} --moe-permute-padding-for-fp8 "
        fi

        if [ ${SCALING_AWARE_TRANSPOSE:-true} = true ]; then  # vs. double quant
            pr_options=" ${pr_options} --moe-scaling-aware-transpose "
        fi
    fi
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

te_options=" \
        --transformer-impl transformer_engine"

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

uneven_split_option=""
if [[ ${MP_PP0_LAYERS:-0} -gt 0 ]]; then
    uneven_split_option="${uneven_split_option} \
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
    "
fi
if [[ ${MP_PPN_LAYERS:-0} -gt 0 ]]; then
    uneven_split_option="${uneven_split_option} \
        --decoder-last-pipeline-num-layers ${MP_PPN_LAYERS}
    "
fi

#动态bs
enable_rampup_bs=false
if  [[ $enable_rampup_bs = false ]]; then
    TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    trainrampup_option="\
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} "
else
    warm_step=2000
    GLOBAL_BATCH_SIZE_avg=5840
    TRAIN_SAMPLES=$(( ${TRAIN_TOKENS} / ${SEQ_LEN} ))
    LR_WARMUP_SAMPLES=$((${warm_step} * ${GLOBAL_BATCH_SIZE_avg} ))
    LR_DECAY_SAMPLES=$(( ${TRAIN_TOKENS} /  ${SEQ_LEN} ))

    trainrampup_option="\
        --lr-decay-samples ${LR_DECAY_SAMPLES} \
        --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
        --train-samples ${TRAIN_SAMPLES} \
        --rampup-batch-size 1920 960 54931640 "
fi

PREFIX="pretraindsv2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
DATA_CACHE_PATH="/mnt/data/public/tmp"
dataset_option=" \
    --data-path ${DATASET_PATH} \
    --data-cache-path $DATA_CACHE_PATH \
    --num-workers 0 \
    --split 100,0,0"

TIMESTAMP=$(date "+%Y%m%d-%H%M")
NAME="${PREFIX}-pr-${PR}-pp-${PP}-ep-${EP}-ac-${AC}_${DLC_JOB_ID:-${TIMESTAMP}}"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES \
    --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    --tee 3 --log_dir ${OUTPUT_DIR}/logs/${NAME}"

PREFIX="DS-v2-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
NAME="${PREFIX}-pr-${PR}-pp-${PP}-ac-${AC}"
# NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/data_cache/"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoints/"
mkdir -p "${OUTPUT_BASEPATH}/logs/"
current_time=$(date "+%Y.%m.%d-%H.%M")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoints/${NAME}"

if [ -d "${SAVED_PRETRAIN_CHECKPOINT_PATH}" ]; then
    LATEST_CKPT_FILE=$(ls -1t "${SAVED_PRETRAIN_CHECKPOINT_PATH}"/iter* 2>/dev/null | head -n 1)
    if [ -n "${LATEST_CKPT_FILE}" ]; then
        echo "Found a checkpoint file: ${LATEST_CKPT_FILE}"
        load_options=" \
                    --load ${SAVED_PRETRAIN_CHECKPOINT_PATH}"
    else
        echo "No checkpoint file found in ${SAVED_PRETRAIN_CHECKPOINT_PATH}."
    fi
fi

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std ${INIT_METHOD_STD} \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --log-interval 1 \
        --record-memory-history \
        --log-throughput \
        --eval-interval 100000000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --tokenizer-type ${TOKENIZER_TYPE:-021Tokenizer} \
        --tokenizer-model $TOKENIZER_PATH \
        --vocab-file $TOKENIZER_PATH/tokenizer.model \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --no-bias-swiglu-fusion \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --rotary-base ${ROPE_THETA} \
        --rotary-scaling-factor ${SCALE_FACTOR} \
        --rotary-seq-len-interpolation-factor 1 \
        --kv-channels ${V_HEAD_DIM} \
        --qk-layernorm \
        --moe-router-dtype fp32 \
        --moe-permute-fusion \
        --auto-detect-ckpt-format \
        --legacy-tokenizer \
        --moe-router-fusion \
        --multi-latent-attention"

if [[ ${SEQWARM:-off} = on ]]; then
seqwarm_options=" --warmup-seq-length 0:2048,100:4096 "
fi

# User Optimizer CPU Offloading
if [[ ${OFFLOAD_OPTIMIZER:-false} = true ]] ; then
    new_options=" ${new_options} --optimizer-cpu-offload --use-precision-aware-optimizer \
        --main-grads-dtype bf16 "
fi

# Precision Aware Optimizer
PAO_LEVEL=${PAO:-none}
if [[ $PAO_LEVEL = none ]]; then
    new_options=" ${new_options} \
    "
elif [[ $PAO_LEVEL = moments ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
    "
elif [[ $PAO_LEVEL = grads ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
        --main-grads-dtype bf16 \
    "
elif [[ $PAO_LEVEL = weights ]]; then
    new_options=" ${new_options} \
        --use-precision-aware-optimizer \
        --exp-avg-dtype fp16 \
        --exp-avg-sq-dtype fp16 \
        --main-grads-dtype bf16 \
        --main-params-dtype fp16 \
    "
else
    echo "PAO_LEVEL=${PAO_LEVEL} is not a valid option. Valid options include: none, moments, grads, weights"
    exit 1
fi                

if [[ ${MANUAL_GC:-0} -gt 0 ]]; then
    megatron_options=" ${megatron_options} --manual-gc --manual-gc-interval ${MANUAL_GC} "
fi

if [[ ${OPTIMIZER:-adam} = muon ]]; then
    optimizer_options=" --optimizer muon \
        --muon-matched-adamw-rms 0.2 \
        --apply-muon-qk-clip "
fi

run_cmd="torchrun $DISTRIBUTED_ARGS ${MEGATRON_PATH}/pretrain_gpt.py
 ${megatron_options} ${trainrampup_option} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${fl_options} ${sp_options} ${moe_options} ${offload_option} ${sft_option} ${vp_options} \
 ${uneven_split_option} ${prof_options} ${seqwarm_options} ${new_options} ${fsdp_options} ${ckpt_options} ${optimizer_options}"

echo ${run_cmd}
[[ $RANK = 0 ]] && mkdir -p ${OUTPUT_DIR}/logs/${NAME} && echo ${run_cmd} > ${OUTPUT_DIR}/logs/${NAME}/${MODEL_SIZE}-pp-${PP}-ep-${EP}-AC-${AC}-gbs-${GLOBAL_BATCH_SIZE}_${current_time}-cmd.sh
eval ${run_cmd}
