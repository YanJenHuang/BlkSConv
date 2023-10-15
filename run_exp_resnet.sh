arch=${1}
round=${2}

log_root="./experiments-resnet/experiments_logs"
arrINFO=(${arch//_/ }) # cifar10_resnet20_blksconv-HSA+V50M50P50b
db_name=${arrINFO[0]}


echo "Running ${arch}_r${round}"
python scripts/resnet_trainer.py --db_name ${db_name} --arch ${arch} --round ${round}|& tee -a ${log_root}/${db_name}/${arch}_r${round}.log

# Example:
# bash run_exp_resnet.sh imagenet_resnet18 1
# bash run_exp_resnet.sh imagenet_resnet18_blksonv-HSA+V50M50P50s 1
# bash run_exp_resnet.sh cifar10_resnet20_blksconv-HSA+V50M50P50b 1
# bash run_exp_resnet.sh cifar100_resnet56_blksconv-HSA+V50M75P75s 2
# bash run_exp_resnet.sh dogs_resnet18_blksconv-HSA+V50M50P50s 3
# bash run_exp_resnet.sh flowers_resnet18_blksconv-HSA+V50M50P50s 4 