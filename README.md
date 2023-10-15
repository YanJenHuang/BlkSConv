# Block-wise Separable Convolutions

Run the "run_exp_resnet.sh" script to train a specific model:

## Train the model

0. Create experimental folders:
```
python create_experiments_folder.py
```

1. Run the standard ResNet
```
bash run_exp_resnet.sh {$db_name}_{$resnet_backbone}
```

2. Run the BlkSConv-based ResNet
```
bash run_exp_resnet.sh {$db_name}_{$resnet_backbone}_blksconv-HSA+{$hyperparameter} {$exp_round}
```

## Arguments

see all the available architectures in ./scripts/resnet.py line 282,
you can also use the scripts/HSA_playground.ipynb to search for other BlkSConv-based model.

``` 
{$db_name}: imagenet, cifar10, cifar100, dogs, flowers
```
```
{$resnet_backbone}: 
resnet10, resnet18, resnet26 (for imagenet, dogs, and flowers)
resnet20, resnet56 (for cifar10 and cifar100)
```
```
{$hyperparameter}: 
V50M50P50s, variance threshold 0.5, MAdds threshold 0.5, Parameter threshold 0.5, with selection strategy small.
V50M75P50b, variance threshold 0.5, MAdds threshold 0.75, Parameter threshold 0.5, with selection strategy big.
```

## Experimental examples
```
bash run_exp_resnet.sh flowers_resnet18 1
bash run_exp_resnet.sh flowers_resnet18_blksconv-HSA+V50M50P50s 1

bash run_exp_resnet.sh dogs_resnet18 1
bash run_exp_resnet.sh dogs_resnet18_blksconv-HSA+V50M75P50b 1

bash run_exp_resnet.sh cifar10_resnet20 1
bash run_exp_resnet.sh cifar10_resnet20_blksconv-HSA+V50M75P50b 1

bash run_exp_resnet.sh imagenet_resnet18 1
bash run_exp_resnet.sh imagenet_resnet18_blksconv-HSA+V50M50P50s 1
```

## Use the hyperparameter search algorithm to find the BlkSConv-based architecture

Run the jupyter notebook ./scripts/HSA_playground.ipynb

We have provide the "imagenet_resnet18_r1.pth" checkpoint file in ```./experiments-resnet/experiments_save_ckpt/imagenet/```