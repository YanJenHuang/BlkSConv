import torch.nn
import torch.nn.init

from common import Classifier
from common import get_activation as get_activation

from conv_block import conv1x1_block, conv3x3_block, conv7x7_block
from blksconv_block import blksconv


###
#%% ResNet building blocks
###

class InitUnitLarge(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv = conv7x7_block(in_channels=in_channels, out_channels=out_channels, stride=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class InitUnitSmall(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 preact=False):
        super().__init__()

        self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                use_bn=not preact,
                activation=None if preact else "relu")

    def forward(self, x):
        x = self.conv(x)
        return x


class PostActivation(torch.nn.Module):
    def __init__(self,
                 channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x


class StandardUnit(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=out_channels, out_channels=out_channels, stride=1, activation=None)
        if self.use_projection:
            self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride, activation=None)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_projection:
            residual = self.projection(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)
        return x

class PreactUnit(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        self.bn = torch.nn.BatchNorm2d(num_features=in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=out_channels, out_channels=out_channels, use_bn=False, activation=None)
        if self.use_projection:
            self.projection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride, use_bn=False, activation=None)

    def forward(self, x):
        if self.use_projection:
            x = self.bn(x)
            x = self.relu(x)
            residual = self.projection(x)
        else:
            residual = x
            x = self.bn(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x


class ResNet(torch.nn.Module):
    def __init__(self,
                 channels,
                 num_classes,
                 preact=False,
                 init_unit_channels=64,
                 use_init_unit_type='Large', # 'Large', 'Small'
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.use_init_unit_type = use_init_unit_type
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init unit
        if self.use_init_unit_type == 'Large':
            self.backbone.add_module("init_unit", InitUnitLarge(in_channels=in_channels, out_channels=init_unit_channels))
        elif self.use_init_unit_type == 'Small':
            self.backbone.add_module("init_unit", InitUnitSmall(in_channels=in_channels, out_channels=init_unit_channels, preact=preact))
        else:
            raise ValueError('use_init_unit_type: {}'.format(self.use_init_unit_type))

        # stages
        in_channels = init_unit_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = 2 if (unit_id == 0) and (stage_id != 0) else 1
                if preact:
                    stage.add_module("unit{}".format(unit_id + 1), PreactUnit(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                else:
                    stage.add_module("unit{}".format(unit_id + 1), StandardUnit(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        if preact:
            self.backbone.add_module("final_activation", PostActivation(in_channels))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))

        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

###
#%% model definitions
###

def build_resnet(num_classes,
                 units_per_stage,
                 width_multiplier = 1.0,
                 preact = False,
                 db_resolution = 'small'):
    if db_resolution=='small':
        # todo: check whether init_channels have to be scaled based on width_factor (guess resnet: yes; wrn: no)
        init_unit_channels = 16
        channels_in_stage = [16, 32, 64]
        channels = [[int(ch * width_multiplier)] * rep for (ch, rep) in zip(channels_in_stage, units_per_stage)]
        use_init_unit_type = 'Small'
        in_size = (32, 32)
    else:
        init_unit_channels = 64
        channels_in_stage = [64, 128, 256, 512]
        channels = [[int(ch * width_multiplier)] * rep for (ch, rep) in zip(channels_in_stage, units_per_stage)]
        use_init_unit_type = 'Large'
        in_size = (224, 224)

    net = ResNet(channels=channels,
                 num_classes=num_classes,
                 preact=preact,
                 init_unit_channels=init_unit_channels,
                 use_init_unit_type=use_init_unit_type,
                 in_size=in_size)
    return net


def get_resnet(architecture, num_classes):
    print(architecture)
    db_resolution_type, backbone = architecture.split('_')
    if 'cifar' in db_resolution_type:
        backbone = f'cifar_{backbone}'

    units_per_stage = {
        "resnet10": [1, 1, 1, 1],
        "resnet18": [2, 2, 2, 2],
        "resnet26": [3, 3, 3, 3],

        "cifar_resnet20": [3, 3, 3],
        "cifar_resnet56": [9, 9, 9],
    }

    width_multiplier = 1.0

    if 'cifar' in architecture:
        db_resolution='small'
    else:
        db_resolution='large'

    # build model
    model = build_resnet(
        num_classes=num_classes,
        units_per_stage=units_per_stage[backbone],
        width_multiplier=width_multiplier,
        preact=False,
        db_resolution=db_resolution,
    )

    return model

# replacer nn.Conv to BlkSConv
def resnet_blksconv_replacer(model, unit_per_stage, stage_replace_flag, replace_type='s1t1', backbone=None):
    # unit_per_stage = [3,3,3,3]
    # stage_replace_flag = [False,True,True,True]
    def replace_standardunit_idx_list(unit_per_stage, stage_replace_flag):
        replace_idx_list=[]
        for num_unit, replace_flag in zip(unit_per_stage, stage_replace_flag):
            for u in range(num_unit):
                replace_idx_list.append(replace_flag)
        return replace_idx_list
    replace_idx_list = replace_standardunit_idx_list(unit_per_stage, stage_replace_flag)

    # s: number of basis vector
    # t: number of block_depth
    replace_type_list = [
        's1t1', 's1t2', 's1t4',
        's2t1', 's2t2', 's2t4',
        's3t1', 's3t2', 's3t4',
        's4t1', 's4t2', 's4t4',
        's5t1', 's5t2', 's5t4',
        's6t1', 's6t2', 's6t4',
        's7t1', 's7t2', 's7t4',
        's8t1', 's8t2', 's8t4',
        's9t1', 's9t2', 's9t4',
    ]

    """
    Hyperparameters Search Algorithm (HSA):
    The architectures search by HSA
        large-scale:
            imagenet: resnet10, resnet18, resnet26
        fine-grained:
            dogs: resnet10, resnet18, resnet26
            flowers: resnet10, resnet18, resnet26
        small-scale:
            cifar10/100: resnet20, resnet56
    """
    HSA_dict = {
        # imagenet-resnet10-HSA
        "imagenet_resnet10_HSA+V50M50P50b": [[False], ['s1t1-s4t1'],['s1t1-s4t1'],['s1t1-s4t1']],
        "imagenet_resnet10_HSA+V50M50P75b": [[False], ['s1t1-s4t1'],['s1t1-s4t1'],['s1t1-s4t1']],
        "imagenet_resnet10_HSA+V50M75P50b": [[False], ['s1t1-s4t1'],['s1t1-s4t1'],['s1t1-s4t1']],
        "imagenet_resnet10_HSA+V50M75P75b": [[False], ['s1t1-s6t1'],['s1t1-s6t1'],['s1t1-s6t1']],
        "imagenet_resnet10_HSA+V75M50P50b": [[False], ['conv-s4t1'],['conv-s4t1'],['conv-s4t1']],
        "imagenet_resnet10_HSA+V75M50P75b": [[False], ['conv-s4t1'],['conv-s4t1'],['conv-s4t1']],
        "imagenet_resnet10_HSA+V75M75P50b": [[False], ['conv-s4t1'],['conv-s4t1'],['conv-s4t1']],
        "imagenet_resnet10_HSA+V75M75P75b": [[False], ['conv-s6t1'],['conv-s6t1'],['conv-s6t1']],
        "imagenet_resnet10_HSA+V50M50P50s": [[False], ['s1t1-s1t1'],['s1t1-s2t2'],['s1t1-s1t2']],
        "imagenet_resnet10_HSA+V50M50P75s": [[False], ['s1t1-s1t1'],['s1t1-s2t2'],['s1t1-s1t2']],
        "imagenet_resnet10_HSA+V50M75P50s": [[False], ['s1t1-s1t1'],['s1t1-s2t2'],['s1t1-s1t2']],
        "imagenet_resnet10_HSA+V50M75P75s": [[False], ['s1t1-s1t1'],['s1t1-s2t2'],['s1t1-s1t2']],
        "imagenet_resnet10_HSA+V75M50P50s": [[False], ['conv-s3t1'],['conv-s3t1'],['conv-s1t1']],
        "imagenet_resnet10_HSA+V75M50P75s": [[False], ['conv-s3t1'],['conv-s3t1'],['conv-s1t1']],
        "imagenet_resnet10_HSA+V75M75P50s": [[False], ['conv-s5t2'],['conv-s5t2'],['conv-s1t1']],
        "imagenet_resnet10_HSA+V75M75P75s": [[False], ['conv-s5t2'],['conv-s5t2'],['conv-s1t1']],

        # imagenet-resnet18-HSA
        "imagenet_resnet18_HSA+V50M50P50b": [[False, False], ['s1t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet18_HSA+V50M50P75b": [[False, False], ['s1t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet18_HSA+V50M75P50b": [[False, False], ['s1t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet18_HSA+V50M75P75b": [[False, False], ['s1t1-s6t1','s6t1-s6t1'],['s1t1-s6t1','s6t1-s6t1'],['s1t1-s6t1','s6t1-s6t1']],
        "imagenet_resnet18_HSA+V75M50P50b": [[False, False], ['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1']],
        "imagenet_resnet18_HSA+V75M50P75b": [[False, False], ['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1']],
        "imagenet_resnet18_HSA+V75M75P50b": [[False, False], ['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1']],
        "imagenet_resnet18_HSA+V75M75P75b": [[False, False], ['conv-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1']],
        "imagenet_resnet18_HSA+V50M50P50s": [[False, False], ['s1t1-s1t1','s2t2-s3t2'],['s1t1-s1t1','s3t2-s3t2'],['s1t1-s3t2','s1t1-s1t2']],
        "imagenet_resnet18_HSA+V50M50P75s": [[False, False], ['s1t1-s1t1','s2t2-s3t2'],['s1t1-s1t1','s3t2-s3t2'],['s1t1-s3t2','s1t1-s1t2']],
        "imagenet_resnet18_HSA+V50M75P50s": [[False, False], ['s1t1-s1t1','s2t2-s3t2'],['s1t1-s1t1','s4t4-s4t4'],['s1t1-s3t2','s1t1-s1t2']],
        "imagenet_resnet18_HSA+V50M75P75s": [[False, False], ['s1t1-s1t1','s2t2-s3t2'],['s1t1-s1t1','s4t4-s4t4'],['s1t1-s3t2','s1t1-s1t2']],
        "imagenet_resnet18_HSA+V75M50P50s": [[False, False], ['conv-s3t1','s3t1-s4t1'],['conv-s3t1','s3t1-s3t1'],['conv-s4t1','s2t1-s3t4']],
        "imagenet_resnet18_HSA+V75M50P75s": [[False, False], ['conv-s3t1','s3t1-s4t1'],['conv-s3t1','s3t1-s3t1'],['conv-s4t1','s2t1-s3t4']],
        "imagenet_resnet18_HSA+V75M75P50s": [[False, False], ['conv-s5t2','s5t2-s4t1'],['conv-s5t2','s3t1-s3t1'],['conv-s6t2','s2t1-s3t4']],
        "imagenet_resnet18_HSA+V75M75P75s": [[False, False], ['conv-s5t2','s5t2-s4t1'],['conv-s5t2','s3t1-s3t1'],['conv-s6t2','s2t1-s3t4']],

        # imagenet-resnet26-HSA
        "imagenet_resnet26_HSA+V50M50P50b": [[False, False, False], ['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet26_HSA+V50M50P75b": [[False, False, False], ['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet26_HSA+V50M75P50b": [[False, False, False], ['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1'],['s1t1-s4t1','s4t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet26_HSA+V50M75P75b": [[False, False, False], ['s1t1-s6t1','s6t1-s6t1','s6t1-s6t1'],['s1t1-s6t1','s6t1-s6t1','s6t1-s6t1'],['s1t1-s6t1','s6t1-s6t1','s6t1-s6t1']],
        "imagenet_resnet26_HSA+V75M50P50b": [[False, False, False], ['conv-s4t1','s4t1-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet26_HSA+V75M50P75b": [[False, False, False], ['conv-s4t1','s4t1-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet26_HSA+V75M75P50b": [[False, False, False], ['conv-s4t1','s4t1-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1','s4t1-s4t1']],
        "imagenet_resnet26_HSA+V75M75P75b": [[False, False, False], ['conv-s6t1','s6t1-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1','s6t1-s6t1']],
        "imagenet_resnet26_HSA+V50M50P50s": [[False, False, False], ['s1t1-s1t1','s1t1-s3t2','s3t2-s3t2'],['s1t1-s1t1','s1t1-s3t2','s3t2-s3t2'],['s1t1-s3t2','s1t1-s1t1','s3t4-s1t2']],
        "imagenet_resnet26_HSA+V50M50P75s": [[False, False, False], ['s1t1-s1t1','s1t1-s3t2','s3t2-s3t2'],['s1t1-s1t1','s1t1-s3t2','s3t2-s3t2'],['s1t1-s3t2','s1t1-s1t1','s3t4-s1t2']],
        "imagenet_resnet26_HSA+V50M75P50s": [[False, False, False], ['s1t1-s1t1','s1t1-s3t2','s3t2-s3t2'],['s1t1-s1t1','s1t1-s4t4','s3t2-s3t2'],['s1t1-s5t4','s1t1-s1t1','s3t4-s1t2']],
        "imagenet_resnet26_HSA+V50M75P75s": [[False, False, False], ['s1t1-s1t1','s1t1-s3t2','s3t2-s3t2'],['s1t1-s1t1','s1t1-s4t4','s3t2-s3t2'],['s1t1-s5t4','s1t1-s1t1','s3t4-s1t2']],
        "imagenet_resnet26_HSA+V75M50P50s": [[False, False, False], ['conv-s3t1','s3t1-s4t1','s4t1-s4t1'],['conv-s3t1','s3t1-s3t1','s3t1-s3t1'],['conv-s3t1','s4t2-s2t1','s3t2-s3t4']],
        "imagenet_resnet26_HSA+V75M50P75s": [[False, False, False], ['conv-s3t1','s3t1-s4t1','s4t1-s4t1'],['conv-s3t1','s3t1-s3t1','s3t1-s3t1'],['conv-s3t1','s4t2-s2t1','s3t2-s3t4']],
        "imagenet_resnet26_HSA+V75M75P50s": [[False, False, False], ['conv-s5t2','s5t2-s4t1','s4t1-s4t1'],['conv-s5t2','s5t2-s3t1','s3t1-s3t1'],['conv-s3t1','s4t2-s2t1','s3t2-s3t4']],
        "imagenet_resnet26_HSA+V75M75P75s": [[False, False, False], ['conv-s5t2','s5t2-s4t1','s4t1-s4t1'],['conv-s5t2','s5t2-s3t1','s3t1-s3t1'],['conv-s3t1','s4t2-s2t1','s3t2-s3t4']],

        # cifar10-resnet20-HSA
        "cifar10_resnet20_HSA+V50M50P50b": [[False, False, False], [False, False, False], ['s1t1-s3t1','s3t1-s3t1','s3t1-s3t1']],
        "cifar10_resnet20_HSA+V50M50P75b": [[False, False, False], [False, False, False], ['s1t1-s3t1','s3t1-s3t1','s3t1-s3t1']],
        "cifar10_resnet20_HSA+V50M75P50b": [[False, False, False], [False, False, False], ['s1t1-s5t2','s5t2-s5t2','s5t2-s5t2']],
        "cifar10_resnet20_HSA+V50M75P75b": [[False, False, False], [False, False, False], ['s1t1-s5t1','s5t1-s5t1','s5t1-s5t1']],
        "cifar10_resnet20_HSA+V75M50P50b": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s3t1']],
        "cifar10_resnet20_HSA+V75M50P75b": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s3t1']],
        "cifar10_resnet20_HSA+V75M75P50b": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s5t2-s5t2']],
        "cifar10_resnet20_HSA+V75M75P75b": [[False, False, False], [False, False, False], ['conv-s5t1','s5t1-s5t1','s5t1-s5t1']],
        "cifar10_resnet20_HSA+V50M50P50s": [[False, False, False], [False, False, False], ['s1t1-s3t1','s3t1-s2t1','s2t1-s1t1']],
        "cifar10_resnet20_HSA+V50M50P75s": [[False, False, False], [False, False, False], ['s1t1-s3t1','s3t1-s2t1','s2t1-s1t1']],
        "cifar10_resnet20_HSA+V50M75P50s": [[False, False, False], [False, False, False], ['s1t1-s4t2','s4t2-s2t1','s2t1-s1t1']],
        "cifar10_resnet20_HSA+V50M75P75s": [[False, False, False], [False, False, False], ['s1t1-s4t2','s4t2-s2t1','s2t1-s1t1']],
        "cifar10_resnet20_HSA+V75M50P50s": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s1t1']],
        "cifar10_resnet20_HSA+V75M50P75s": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s1t1']],
        "cifar10_resnet20_HSA+V75M75P50s": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s1t1']],
        "cifar10_resnet20_HSA+V75M75P75s": [[False, False, False], [False, False, False], ['conv-s4t1','s4t1-s4t1','s3t1-s1t1']],

        # cifar10-resnet56-HSA
        "cifar10_resnet56_HSA+V50M50P50b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1',]],
        "cifar10_resnet56_HSA+V50M50P75b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1',]],
        "cifar10_resnet56_HSA+V50M75P50b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2',]],
        "cifar10_resnet56_HSA+V50M75P75b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1',]],
        "cifar10_resnet56_HSA+V75M50P50b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','conv-s3t1','s3t1-s3t1',]],
        "cifar10_resnet56_HSA+V75M50P75b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','conv-s3t1','s3t1-s3t1',]],
        "cifar10_resnet56_HSA+V75M75P50b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s5t2','conv-s5t2','conv-s5t2','s5t2-s5t2',]],
        "cifar10_resnet56_HSA+V75M75P75b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1',]],
        "cifar10_resnet56_HSA+V50M50P50s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s1t1-s1t2',]],
        "cifar10_resnet56_HSA+V50M50P75s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s1t1-s1t2',]],
        "cifar10_resnet56_HSA+V50M75P50s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s1t1-s1t2',]],
        "cifar10_resnet56_HSA+V50M75P75s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s1t1-s1t2',]],
        "cifar10_resnet56_HSA+V75M50P50s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','conv-s3t1','s3t1-s1t1',]],
        "cifar10_resnet56_HSA+V75M50P75s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','conv-s3t1','s3t1-s1t1',]],
        "cifar10_resnet56_HSA+V75M75P50s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','conv-s3t1','s4t2-s1t1',]],
        "cifar10_resnet56_HSA+V75M75P75s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-s4t1','s4t1-s4t1','s4t1-s4t1','s4t1-s4t1','s4t1-s4t1','s4t1-s3t1','s4t1-s3t1','s4t1-s3t1','s4t2-s1t1',]],

        # cifar100-resnet20-HSA
        "cifar100_resnet20_HSA+V50M50P50b": [[False, False, False], [False, False, False], ['s1t1-s3t1','s3t1-s3t1','s3t1-s3t1']],
        "cifar100_resnet20_HSA+V50M50P75b": [[False, False, False], [False, False, False], ['s1t1-s3t1','s3t1-s3t1','s3t1-s3t1']],
        "cifar100_resnet20_HSA+V50M75P50b": [[False, False, False], [False, False, False], ['s1t1-s5t2','s5t2-s5t2','s5t2-s5t2']],
        "cifar100_resnet20_HSA+V50M75P75b": [[False, False, False], [False, False, False], ['s1t1-s5t1','s5t1-s5t1','s5t1-s5t1']],
        "cifar100_resnet20_HSA+V75M50P50b": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s3t1']],
        "cifar100_resnet20_HSA+V75M50P75b": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s3t1']],
        "cifar100_resnet20_HSA+V75M75P50b": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s5t2-s5t2']],
        "cifar100_resnet20_HSA+V75M75P75b": [[False, False, False], [False, False, False], ['conv-s5t1','s5t1-s5t1','s5t1-s5t1']],
        "cifar100_resnet20_HSA+V50M50P50s": [[False, False, False], [False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s1t1']],
        "cifar100_resnet20_HSA+V50M50P75s": [[False, False, False], [False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s1t1']],
        "cifar100_resnet20_HSA+V50M75P50s": [[False, False, False], [False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s1t1']],
        "cifar100_resnet20_HSA+V50M75P75s": [[False, False, False], [False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s1t1']],
        "cifar100_resnet20_HSA+V75M50P50s": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s2t1']],
        "cifar100_resnet20_HSA+V75M50P75s": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s2t1']],
        "cifar100_resnet20_HSA+V75M75P50s": [[False, False, False], [False, False, False], ['conv-conv','conv-conv','s3t1-s2t1']],
        "cifar100_resnet20_HSA+V75M75P75s": [[False, False, False], [False, False, False], ['conv-s4t1','s4t1-s4t1','s3t1-s2t1']],

        # cifar100-resnet56-HSA
        "cifar100_resnet56_HSA+V50M50P50b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1']],
        "cifar100_resnet56_HSA+V50M50P75b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1','s3t1-s3t1']],
        "cifar100_resnet56_HSA+V50M75P50b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2','s5t2-s5t2']],
        "cifar100_resnet56_HSA+V50M75P75b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1']],
        "cifar100_resnet56_HSA+V75M50P50b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','s3t1-s3t1']],
        "cifar100_resnet56_HSA+V75M50P75b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','s3t1-s3t1']],
        "cifar100_resnet56_HSA+V75M75P50b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s5t2','s5t2-s5t2','s5t2-s5t2']],
        "cifar100_resnet56_HSA+V75M75P75b": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1','s5t1-s5t1']],
        "cifar100_resnet56_HSA+V50M50P50s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t2','s2t2-s1t2']],
        "cifar100_resnet56_HSA+V50M50P75s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t2','s2t2-s1t2']],
        "cifar100_resnet56_HSA+V50M75P50s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t2','s2t2-s1t2']],
        "cifar100_resnet56_HSA+V50M75P75s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['s1t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t1','s2t1-s2t2','s2t2-s1t2']],
        "cifar100_resnet56_HSA+V75M50P50s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','s3t1-s1t1']],
        "cifar100_resnet56_HSA+V75M50P75s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','conv-s3t1','s3t1-s1t1']],
        "cifar100_resnet56_HSA+V75M75P50s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-conv','conv-s3t1','s5t2-s4t2','s4t2-s1t1']],
        "cifar100_resnet56_HSA+V75M75P75s": [[False, False, False, False, False, False, False, False, False], [False, False, False, False, False, False, False, False, False], ['conv-s4t1','s4t1-s4t1','s4t1-s4t1','s4t1-s4t1','s4t1-s4t1','s4t1-s4t1','s4t1-s3t1','s5t2-s4t2','s4t2-s1t1']],

        # dogs-resnet18-HSA
        "dogs_resnet18_HSA+V50M50P50b": [[False, False], ['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1']],
        "dogs_resnet18_HSA+V50M50P75b": [[False, False], ['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1']],
        "dogs_resnet18_HSA+V50M75P50b": [[False, False], ['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1']],
        "dogs_resnet18_HSA+V50M75P75b": [[False, False], ['conv-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1']],
        "dogs_resnet18_HSA+V75M50P50b": [[False, False], ['conv-conv','s4t1-s4t1'],['conv-conv','conv-conv'],['conv-conv','s4t1-s4t1']],
        "dogs_resnet18_HSA+V75M50P75b": [[False, False], ['conv-conv','s4t1-s4t1'],['conv-conv','conv-conv'],['conv-conv','s4t1-s4t1']],
        "dogs_resnet18_HSA+V75M75P50b": [[False, False], ['conv-conv','s4t1-s4t1'],['conv-conv','conv-conv'],['conv-conv','s4t1-s4t1']],
        "dogs_resnet18_HSA+V75M75P75b": [[False, False], ['conv-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1']],
        "dogs_resnet18_HSA+V50M50P50s": [[False, False], ['conv-s3t1','s2t1-s3t2'],['conv-s2t1','s2t1-s2t1'],['conv-s3t1','s3t2-s3t4']],
        "dogs_resnet18_HSA+V50M50P75s": [[False, False], ['conv-s3t1','s2t1-s3t2'],['conv-s2t1','s2t1-s2t1'],['conv-s3t1','s3t2-s3t4']],
        "dogs_resnet18_HSA+V50M75P50s": [[False, False], ['conv-s4t2','s2t1-s3t2'],['conv-s2t1','s2t1-s2t1'],['conv-s5t2','s5t4-s3t4']],
        "dogs_resnet18_HSA+V50M75P75s": [[False, False], ['conv-s4t2','s2t1-s3t2'],['conv-s2t1','s2t1-s2t1'],['conv-s5t2','s5t4-s3t4']],
        "dogs_resnet18_HSA+V75M50P50s": [[False, False], ['conv-conv','s4t1-s4t1'],['conv-conv','conv-conv'],['conv-conv','s4t1-s3t1']],
        "dogs_resnet18_HSA+V75M50P75s": [[False, False], ['conv-conv','s4t1-s4t1'],['conv-conv','conv-conv'],['conv-conv','s4t1-s3t1']],
        "dogs_resnet18_HSA+V75M75P50s": [[False, False], ['conv-conv','s4t1-s4t1'],['conv-conv','conv-conv'],['conv-conv','s6t2-s5t2']],
        "dogs_resnet18_HSA+V75M75P75s": [[False, False], ['conv-s5t1','s4t1-s4t1'],['conv-s5t1','s5t1-s5t1'],['conv-s5t1','s6t2-s5t2']],
        
        # dogs-resnet18-HSA (ablation study)
        "dogs_resnet18_HSA+V00M75P75s": [[False, False], ['s1t2-s1t4','s1t4-s1t4'],['s1t4-s1t4','s1t4-s1t4'],['s1t4-s1t4','s1t4-s1t4']],
        "dogs_resnet18_HSA+V10M75P75s": [[False, False], ['s1t2-s1t4','s1t4-s1t4'],['s1t4-s1t4','s1t4-s1t4'],['s1t4-s1t2','s1t4-s1t4']],
        "dogs_resnet18_HSA+V20M75P75s": [[False, False], ['s1t2-s2t4','s1t2-s1t2'],['s1t2-s2t4','s2t4-s2t4'],['s1t1-s3t4','s1t2-s1t4']],
        "dogs_resnet18_HSA+V30M75P75s": [[False, False], ['s1t1-s1t1','s1t1-s2t4'],['s1t1-s1t1','s1t1-s1t1'],['s1t1-s4t4','s3t4-s1t2']],
        "dogs_resnet18_HSA+V40M75P75s": [[False, False], ['conv-s3t2','s3t2-s2t2'],['conv-s3t2','s3t2-s4t4'],['conv-s6t4','s1t1-s3t4']],

        # flowers-resnet18-HSA
        "flowers_resnet18_HSA+V50M50P50b": [[False, False], ['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1']],
        "flowers_resnet18_HSA+V50M75P75b": [[False, False], ['conv-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1'],['conv-s6t1','s6t1-s6t1']],
        "flowers_resnet18_HSA+V50M50P50s": [[False, False], ['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s4t1-s4t1'],['conv-s4t1','s3t1-s3t1']],
        "flowers_resnet18_HSA+V50M75P75s": [[False, False], ['conv-s5t2','s5t2-s5t2'],['conv-s6t2','s4t1-s4t1'],['conv-s4t1','s3t1-s3t1']],

        "flowers_resnet18_HSA+V00M75P75s": [[False, False], ['s1t2-s1t4','s1t4-s1t4'],['s1t4-s1t4','s1t4-s1t4'],['s1t4-s1t4','s1t4-s1t4']],
        "flowers_resnet18_HSA+V10M75P75s": [[False, False], ['s1t2-s1t4','s1t4-s1t4'],['s1t4-s1t2','s1t2-s1t2'],['s1t2-s2t4','s1t2-s1t4']],
        "flowers_resnet18_HSA+V20M75P75s": [[False, False], ['s1t1-s2t4','s2t4-s2t4'],['conv-s1t1','s3t4-s2t2'],['conv-s4t4','s3t4-s3t4']],
        "flowers_resnet18_HSA+V30M75P75s": [[False, False], ['conv-s3t4','s3t4-s3t4'],['conv-s3t2','s5t4-s5t4'],['conv-s6t4','s5t4-s1t1']],
        "flowers_resnet18_HSA+V40M75P75s": [[False, False], ['conv-s4t2','s4t2-s4t2'],['conv-s5t2','s5t2-s5t2'],['conv-s5t2','s4t2-s6t4']],


    }
    if not ((replace_type in replace_type_list) or ('HSA' in replace_type)):
        raise ValueError(f'replace_type {replace_type} error.')

    if replace_type in replace_type_list:
        num_basis = int(replace_type[1])
        blk_depth = int(replace_type[3])
    elif ('HSA' in replace_type) and (backbone is not None):
        backbone = f'{backbone}_{replace_type}'
        print(backbone)
        blksconv_replace_type = HSA_dict[backbone]
        blksconv_replace_type = [item for blkconvlist in blksconv_replace_type for item in blkconvlist]
    else:
        raise ValueError(f'replace_type {replace_type} error.')

    standard_unit_idx = 0
    for module in model.backbone:
        if isinstance(module, torch.nn.Sequential):
            for sub_layer in module:
                if isinstance(sub_layer, StandardUnit):
                    if replace_idx_list[standard_unit_idx]==True:
                        if ('HSA' in replace_type):
                            conv1_type, conv2_type = blksconv_replace_type[standard_unit_idx].split('-')
                            if conv1_type != 'conv':
                                conv1_num_basis = int(conv1_type[1])
                                conv1_blk_depth = int(conv1_type[3])
                                # conv1 replace
                                (in_channels,out_channels,kernel_size,stride,padding) = (sub_layer.conv1.conv.in_channels, sub_layer.conv1.conv.out_channels, sub_layer.conv1.conv.kernel_size, 
                                                                                    sub_layer.conv1.conv.stride, sub_layer.conv1.conv.padding)
                                sub_layer.conv1.conv = blksconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                                    num_basis=conv1_num_basis, blk_depth=conv1_blk_depth,
                                                                    stride=stride, padding=padding, bias=False)
                            if conv2_type != 'conv':
                                conv2_num_basis = int(conv2_type[1])
                                conv2_blk_depth = int(conv2_type[3])
                                # conv2 replace
                                (in_channels,out_channels,kernel_size,stride,padding)  = (sub_layer.conv2.conv.in_channels, sub_layer.conv2.conv.out_channels, sub_layer.conv2.conv.kernel_size, 
                                                                                    sub_layer.conv2.conv.stride, sub_layer.conv2.conv.padding)
                                sub_layer.conv2.conv = blksconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                                    num_basis=conv2_num_basis, blk_depth=conv2_blk_depth,
                                                                    stride=stride, padding=padding, bias=False)
                        else:
                            conv1_num_basis = num_basis
                            conv1_blk_depth = blk_depth
                            conv2_num_basis = num_basis
                            conv2_blk_depth = blk_depth
                            # conv1 replace
                            (in_channels,out_channels,kernel_size,stride,padding) = (sub_layer.conv1.conv.in_channels, sub_layer.conv1.conv.out_channels, sub_layer.conv1.conv.kernel_size, 
                                                                                    sub_layer.conv1.conv.stride, sub_layer.conv1.conv.padding)
                            sub_layer.conv1.conv = blksconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                                num_basis=conv1_num_basis, blk_depth=conv1_blk_depth,
                                                                stride=stride, padding=padding, bias=False)
                            # conv2 replace
                            (in_channels,out_channels,kernel_size,stride,padding)  = (sub_layer.conv2.conv.in_channels, sub_layer.conv2.conv.out_channels, sub_layer.conv2.conv.kernel_size, 
                                                                                    sub_layer.conv2.conv.stride, sub_layer.conv2.conv.padding)
                            sub_layer.conv2.conv = blksconv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                                num_basis=conv2_num_basis, blk_depth=conv2_blk_depth,
                                                                stride=stride, padding=padding, bias=False)
                    standard_unit_idx+=1


def get_resnet_blksconv(architecture, num_classes):

    # ['cifar10_resnet10_blksconv-s1t2',
    #  'cifar100_resnet18_blksconv-HSA+V50M50P50b']
    db_resolution_type, backbone, blksconv_type = architecture.split('_')

    units_per_stage = {
        "resnet10": [1, 1, 1, 1],
        "resnet18": [2, 2, 2, 2],
        "resnet26": [3, 3, 3, 3],

        "cifar_resnet20": [3, 3, 3],
        "cifar_resnet56": [9, 9, 9],
    }
    stage_replace_flag = {
        # large-scale: imagenet
        # fine-grained: dogs, flowers
        "resnet10": [False, True, True, True],
        "resnet18": [False, True, True, True],
        "resnet26": [False, True, True, True],

        # small-scale: cifar
        "cifar_resnet20": [False, False, True],
        "cifar_resnet56": [False, False, True],
    }
    # print(db_resolution_type)
    replace_type=blksconv_type.split('-')[-1] # replace_type='s1t1' or 'HSA+V50M50P50b'
    if 'cifar' in db_resolution_type: 
        backbone = f'cifar_{backbone}'
        model = get_resnet(backbone, num_classes)
        resnet_blksconv_replacer(model, units_per_stage[backbone], stage_replace_flag[backbone], replace_type=replace_type, backbone=backbone.replace('cifar', db_resolution_type))
    elif db_resolution_type in ['imagenet', 'dogs', 'flowers']:
        arch = f'{db_resolution_type}_{backbone}'
        model = get_resnet(arch, num_classes)
        resnet_blksconv_replacer(model, units_per_stage[backbone], stage_replace_flag[backbone], replace_type=replace_type, backbone=arch)

    return model