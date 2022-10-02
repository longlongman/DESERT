#!/bin/bash
DIR=$(dirname $(readlink -f "$0"))
cd $DIR
source bashutil.sh

declare -A BASH_ARGS=(
    # bycha config
    [config]=./configs/training.yaml
    [lib]=shape_pretraining
    [args]=
)
parse_args "$@"

MPIRUN bycha-run \
    --config $config \
    --lib $lib \
    --task.trainer.tensorboard_dir ❗❗❗FILL_THIS❗❗❗ \
    --task.trainer.save_model_dir  ❗❗❗FILL_THIS❗❗❗ \
    --task.trainer.restore_path    ❗❗❗FILL_THIS❗❗❗ \
    ${BASH_ARGS[args]}

