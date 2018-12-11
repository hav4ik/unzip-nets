output_dir="~data/unzip-nets/nova_trio";

# CIFAR
python python/run.py \
    train \
    configs/lenet/one-task-c.json \
    -b 256 \
    -n 100 \
    -k 195 \
    -o $output_dir

# DOGS VS CATS
python python/run.py \
    train \
    configs/lenet/one-task-d.json \
    -b 256 \
    -n 100 \
    -k 195 \
    -o $output_dir

# EMOTIONS
python python/run.py \
    train \
    configs/lenet/one-task-e.json \
    -b 256 \
    -n 100 \
    -k 195 \
    -o $output_dir

# CIFAR + DOGS VS CATS
python python/run.py \
    train \
    configs/lenet/two-tasks-cd.json \
    -b 256 \
    -n 100 \
    -k 195 \
    -o $output_dir

# CIFAR + EMOTIONS
python python/run.py \
    train \
    configs/lenet/two-tasks-ce.json \
    -b 256 \
    -n 100 \
    -k 195 \
    -o $output_dir

# DOGS VS CATS + EMOTIONS
python python/run.py \
    train \
    configs/lenet/two-tasks-de.json \
    -b 256 \
    -n 100 \
    -k 195 \
    -o $output_dir

# CIFAR + DOGS VS CATS + EMOTIONS
python python/run.py \
    train \
    configs/lenet/three-tasks-cde.json \
    -b 256 \
    -n 100 \
    -k 195 \
    -o $output_dir
