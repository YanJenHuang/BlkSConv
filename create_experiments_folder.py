import os

root = './'
experiments_root = './experiments-resnet'

experiments_root_folder_list = ['experiments_best_acc',
                                'experiments_logs',
                                'experiments_save_ckpt',
                                'experiments_tensorboards_acc',
                                'experiments_tensorboards_loss']

db_name_list = ['cifar10',
                'cifar100',
                'dogs',
                'flowers',
                'imagenet']

for root_path in experiments_root_folder_list:
    for db_name in db_name_list:
        DIR = os.path.join(experiments_root, root_path, db_name)
        if not os.path.exists(DIR):
            os.makedirs(DIR)