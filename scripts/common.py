import torch.nn

def get_activation(activation):
    if activation == "relu":
        return torch.nn.ReLU(inplace=True)
    else:
        raise NotImplementedError("Activation {} not implemented".format(activation))

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Classifier(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
                bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

    def init_params(self):
        torch.nn.init.xavier_normal_(self.conv.weight, gain=1.0)