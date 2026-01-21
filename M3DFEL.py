import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange


# SE 模块定义
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class M3DFEL(nn.Module):
    """The proposed M3DFEL framework

    Args:
        args
    """

    def __init__(self, args):
        super(M3DFEL, self).__init__()

        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        # backbone networks
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            *list(model.children())[:-1])  # after avgpool 512x1
        # 将 LSTM 替换为 GRU
        self.gru = nn.GRU(input_size=512, hidden_size=512,
                          num_layers=2, batch_first=True, bidirectional=True)

        # 定义 SE 模块
        self.se_layer = SELayer(channel=1024)

        self.norm = nn.LayerNorm(1024)
        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)

        # classifier
        self.fc = nn.Linear(1024, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)

    def MIL(self, x):
        """The Multi Instance Learning Agregation of instances

        Inputs:
            x: [batch, bag_size, 512]
        """
        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        # [batch, bag_size, 1024]
        ori_x = x

        # 使用 SE 模块替换多头自注意力机制
        x = x.permute(0, 2, 1)  # 调整维度以适应 SE 模块输入 [batch, 1024, bag_size]
        x = self.se_layer(x)
        x = x.permute(0, 2, 1)  # 调整维度回 [batch, bag_size, 1024]

        x = self.norm(x)
        x = torch.sigmoid(x)

        x = ori_x * x

        return x

    def forward(self, x):

        # [batch, 16, 3, 112, 112]
        x = rearrange(x, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                      t1=self.bag_size, t2=self.instance_length)
        # [batch*bag_size, 3, il, 112, 112]

        x = self.features(x).squeeze()
        # [batch*bag_size, 512]
        x = rearrange(x, '(b t) c -> b t c', t=self.bag_size)

        # [batch, bag_size, 512]
        x = self.MIL(x)
        # [batch, bag_size, 1024]

        x = self.pwconv(x).squeeze()
        # [batch, 1024]
        out = self.fc(x)
        # [batch, 7]

        return out
