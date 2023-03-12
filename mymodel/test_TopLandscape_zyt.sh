#!/bin/bash
set -x

source myenv.sh

echo "args: $@" #打印所有参数


# 设置数据集目录
DATADIR=${DATADIR_TopLandscape}

# -z 判断变量是否为空
#两种用法：
#   [ -z "$pid" ] 单对中括号变量必须要加双引号
#   [[ -z $pid ]] 双对括号，变量不用加双引号
#
[[ -z $DATADIR ]] && DATADIR='../datasets/TopLandscape'

# set a comment via `COMMENT` ?
suffix=${COMMENT}
echo "suffix=$suffix"

modelName=$1
extraopts=""
if [[ $modelName == "PN" ]]; then
    # modelopts="../networks/example_ParticleNet.py"
    # modelopts="example_ParticleNet_zyt.py"
    modelopts="./network/example_Top_ParticleNet_zyt.py"
    lr="1e-2"
else
    echo "Invalid model $modelName!"
    exit 1
fi

# "kin" 默认为kin, 只有输入
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="kin"
if [[ "${FEATURE_TYPE}" != "kin" ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

echo "FEATURE_TYPE=$FEATURE_TYPE"


# weaver \
#     --data-train "${DATADIR}/train_file.parquet" \
#     --data-val "${DATADIR}/val_file.parquet" \
#     --data-test "${DATADIR}/test_file.parquet" \
#     --data-config ../data/TopLandscape/top_${FEATURE_TYPE}.yaml --network-config $modelopts \
#     --model-prefix training/TopLandscape/${model}/{auto}${suffix}/net \
#     --num-workers 1 --fetch-step 1 --in-memory \
#     --batch-size 512 --samples-per-epoch $((2400 * 512)) --samples-per-epoch-val $((800 * 512)) --num-epochs 1 --gpus 0 \
#     --start-lr $lr --optimizer ranger --log logs/TopLandscape_${model}_{auto}${suffix}.log --predict-output pred.root \
#     --tensorboard TopLandscape_${FEATURE_TYPE}_${model}${suffix} \
#     ${extraopts} "${@:3}"


# python weaver/train.py \
#     --data-train "${DATADIR}/train_file.parquet" \
#     --data-val "${DATADIR}/val_file.parquet" \
#     --data-test "${DATADIR}/test_file.parquet" \
#     --data-config ../data/TopLandscape/top_${FEATURE_TYPE}.yaml --network-config $modelopts \
#     --model-prefix training/TopLandscape/${model}/{auto}${suffix}/net \
#     --num-workers 1 --fetch-step 1 --in-memory \
#     --batch-size 512 --samples-per-epoch $((2400 * 512)) --samples-per-epoch-val $((800 * 512)) --num-epochs 1 --gpus 0 \
#     --start-lr $lr --optimizer ranger --log logs/TopLandscape_${model}_{auto}${suffix}.log --predict-output pred.root \
#     --tensorboard TopLandscape_${FEATURE_TYPE}_${model}${suffix} \
#     ${extraopts} "${@:3}" 

echo "=======test =========="

# python weaver/train.py --predict --data-test "${DATADIR}/test_file.parquet" \
#  --data-config ../data/TopLandscape/top_${FEATURE_TYPE}.yaml \
#  --network-config $modelopts \
#  --model-prefix ./test/net_best_epoch_state.pt \
#  --gpus 0 --batch-size 512 \
#  --predict-output ./test/output.root \

python weaver/train.py --predict --data-test "${DATADIR}/test_file.parquet" \
 --data-config ../data/TopLandscape/top_${FEATURE_TYPE}.yaml \
 --network-config $modelopts \
 --model-prefix ./test/best_simple/net_best_epoch_state.pt \
 --gpus "" --predict-gpus "" \
 --batch-size 1 \
 --predict-output ./test/output.root \
 # 利用哪个模型进行推理


#
#
# TopLandscape : converted from https://zenodo.org/record/2603256
#
#