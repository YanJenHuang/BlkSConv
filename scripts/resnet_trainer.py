import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import custom_datasets.datasets_stanford_dogs as datasets_stanford_dogs
import custom_datasets.datasets_oxford_flowers as datasets_oxford_flowers

from resnet import get_resnet
from resnet import get_resnet_blksconv


from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='BlkSConv-ResNet')
parser.add_argument('--root', default='./', type=str)
parser.add_argument('--exp_root', default='./experiments-resnet', type=str)
parser.add_argument('--db_name', '-db', default='dogs', choices=['imagenet','cifar10','cifar100','dogs','flowers'], help='training dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cifar10_resnet20_blksconv-HSA+V50M50P50b')
parser.add_argument('--round', '-r', default=0, type=int)

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-ckpt-dir', dest='save_ckpt_dir',
                    help='The directory used to save the trained models',
                    default='experiments_save_ckpt', type=str)
parser.add_argument('--save-acc-dir', dest='save_acc_dir',
                    help='The directory used to save the best accuracy in text file',
                    default='experiments_best_acc', type=str)
# parser.add_argument('--save-every', dest='save_every',
#                     help='Saves checkpoints at every specified number of epochs',
#                     type=int, default=10)
best_prec1 = 0

def count_parameters(model, requires_grad=True):
    if requires_grad == True:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    
    print(type(model), pytorch_total_params)
    return pytorch_total_params

def main():
    global args, best_prec1
    args = parser.parse_args()

    SAVE_CKPT_DIR = os.path.join(args.exp_root, args.save_ckpt_dir, args.db_name)
    # Check the save_ckpt_dir exists or not
    if not os.path.exists(SAVE_CKPT_DIR):
        os.makedirs(SAVE_CKPT_DIR)

    SAVE_ACC_DIR = os.path.join(args.exp_root, args.save_acc_dir, args.db_name)
    # Check the save_acc_dir exists or not
    if not os.path.exists(SAVE_ACC_DIR):
        os.makedirs(SAVE_ACC_DIR)

    if args.db_name == 'imagenet': num_classes = 1000
    if args.db_name == 'cifar10': num_classes = 10
    if args.db_name == 'cifar100': num_classes = 100
    if args.db_name == 'dogs': num_classes = 120
    if args.db_name == 'flowers': num_classes = 102
    
    model_name = args.arch
    if 'blksconv' in args.arch:
        # resnet_blksconv
        print(args.arch)
        model = torch.nn.DataParallel(get_resnet_blksconv(architecture=args.arch, num_classes=num_classes))
    else:
        # resnet
        model = torch.nn.DataParallel(get_resnet(architecture=args.arch, num_classes=num_classes))

    print(model_name)
    print(model)
    # initial tensorboard summary_writer
    log_dir_acc = os.path.join(args.exp_root, 'experiments_tensorboards_acc', args.db_name)
    summary_writer_acc = SummaryWriter(log_dir=log_dir_acc)
    log_dir_loss = os.path.join(args.exp_root, 'experiments_tensorboards_loss', args.db_name)
    summary_writer_loss = SummaryWriter(log_dir=log_dir_loss)

    model.cuda()
    cudnn.benchmark = True

    # small-scale: cifar10 dataloader
    if args.db_name == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./Datasets/cifar/root', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./Datasets/cifar/root', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # small-scale: cifar100 dataloader
    if args.db_name == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./Datasets/cifar/root', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root='./Datasets/cifar/root', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
            ])),
            batch_size=128, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # fine-grained: dogs dataloader
    if args.db_name == 'dogs':
        train_loader = torch.utils.data.DataLoader(
            datasets_stanford_dogs.StanfordDogs(root='./Datasets/ImageNetDogs/', train=True, download=True, transform=transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4),
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            datasets_stanford_dogs.StanfordDogs(root='./Datasets/ImageNetDogs/', train=False, download=False, transform=transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])),
            batch_size=128, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    
    # fine-grained: flowers dataloader
    if args.db_name == 'flowers':
        train_loader = torch.utils.data.DataLoader(
            datasets_oxford_flowers.OxfordFlowers102(root='./Datasets/sub_imagenet/oxford_flowers', train=True, download=True, transform=transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4),
                transforms.ToTensor()
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            datasets_oxford_flowers.OxfordFlowers102(root='./Datasets/sub_imagenet/oxford_flowers', train=False, download=False, transform=transforms.Compose([
                transforms.Resize(size=(256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])),
            batch_size=128, shuffle=True,
            num_workers=args.workers, pin_memory=True)

    # large-scale: imagenet dataloader
    if args.db_name == 'imagenet':
        imagenet_root = './Datasets/imagenet/root'
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        trainset = torchvision.datasets.ImageNet(root=imagenet_root, split="train", transform=train_transform)
        validset = torchvision.datasets.ImageNet(root=imagenet_root, split="val", transform=val_transform)
        
        train_batch_size = 256
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    if args.db_name in ['cifar10', 'cifar100', 'dogs', 'flowers']:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # learning rate
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[100, 150, 180], last_epoch=args.start_epoch - 1)

    if args.db_name in ['imagenet']:
        args.epochs = 100
        args.print_freq = 250
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # learning rate
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[30, 60, 90], last_epoch=args.start_epoch - 1)

    # count parameters
    count_parameters(model, requires_grad=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        # using scheduler learning rate function
        lr_scheduler.step()
        
        # evaluate on validation set
        prec1, loss = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        summary_writer_acc.add_scalars('{}_epoch_acc_r{}'.format(model_name, args.round), {'prec1':prec1,
                                         'best_prec1':best_prec1}, epoch+1)
        summary_writer_loss.add_scalar('{}_epoch_loss_r{}'.format(model_name, args.round), loss, epoch+1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(SAVE_CKPT_DIR,'{}_r{}.pth'.format(model_name, args.round)))
    
    # save best acc and num_parameters
    count_grad = count_parameters(model, requires_grad=True)
    acc_filename = '{}_r{}.txt'.format(model_name, args.round)
    f = open(os.path.join(SAVE_ACC_DIR, acc_filename), 'w')
    f.write('{}, {}, {}'.format(model_name, count_grad, best_prec1))
    f.close()

    print(torch.cuda.memory_summary())
    print('{}: {} parameters'.format(args.arch, count_grad))
    

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
    if is_best and 'ckpt' not in filename:
        torch.save(state, filename.replace('.pth', '_best.pth'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
