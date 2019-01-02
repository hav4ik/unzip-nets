#!/bin/bash

# experiments to run;
# TODO: export the array into a separated config file
CONFIGS=(
    "configs/lenet/one-task-c.json"
    "configs/lenet/one-task-d.json"
    "configs/lenet/one-task-e.json"
    "configs/lenet/two-tasks-cd.json"
    "configs/lenet/two-tasks-ce.json"
    "configs/lenet/two-tasks-de.json"
    "configs/lenet/three-tasks-cde.json"
)

# default values
REPEATNUM=1
OUTPUTDIR="out/"
EPOCHS=100
BATCHSIZE=256
STEPSPEREPOCH=195

# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="${1}"
    case ${key} in
    -n|--repeat)
        REPEATNUM="${2}"
        shift # past argument
        shift # past value
        ;;
    -o|--output)
        OUTPUTDIR="${2}"
        shift # past argument
        shift # past value
        ;;
    -h|--help)
        echo "Show help"
        shift # past argument
        ;;
    *)    # unknown option
        shift # past argument
        ;;
    esac
done

# repeat for all configs
for i in $(seq $REPEATNUM);
do
    for config in ${CONFIGS[@]}
    do
        echo "training for ${config}..."
        python python/run.py \
            train \
            ${config} \
            -b ${BATCHSIZE} \
            -n ${EPOCHS} \
            -k ${STEPSPEREPOCH} \
            -o ${OUTPUTDIR}
    done
done
