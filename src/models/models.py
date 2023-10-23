import torch
from torch import nn
import torch.nn.functional as F
from common import device

norm_dict = dict(instance=nn.LazyInstanceNorm1d, batch=nn.LazyBatchNorm1d)

ONEBYONECHANNELS = 4


class AlexNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size=3,
                 dropRate=0.0, act=F.relu, norm='instance', stride=1):
        super(AlexNetBasicBlock, self).__init__()
        self.norm = norm_dict[norm]()
        self.act = act
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kern_size, stride=1,
                              padding="same", bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv(self.act(self.norm(x)))
        if self.droprate > 0:
            out = F.dropout1d(out, p=self.droprate, training=self.training)
        return out


class AlexNet(nn.Module):
    def __init__(self, conv_channels, kern_sizes, out_shape, inp_channels=1, act=F.gelu,
                 droprate=0.0, norm="instance", stride=1):
        super().__init__()
        self.conv_channels = conv_channels
        self.kern_sizes = kern_sizes
        self.act = act
        self.norm = norm
        self.stride = stride
        self.droprate = droprate
        self.conv_part = self._build_layers()
        self.onebyone_conv = nn.Conv1d(in_channels=conv_channels[-1],
                                       out_channels=ONEBYONECHANNELS, kernel_size=1)
        self.logits = nn.LazyLinear(out_shape)

    def forward(self, x):
        conv_out = self.onebyone_conv(self.conv_part(x))
        flat_features = torch.flatten(self.act(conv_out), start_dim=1)
        if self.droprate > 0.0:
            flat_features = F.dropout(
                flat_features, self.droprate, training=self.training)
        return self.logits(flat_features)

    def _build_layers(self):
        blocks = []
        last_conv_chans = 1
        for conv_chan, ksize in zip(self.conv_channels, self.kern_sizes):
            blocks.append(AlexNetBasicBlock(last_conv_chans, conv_chan, ksize,
                                            dropRate=self.droprate, norm=self.norm))
            last_conv_chans = conv_chan
        return nn.Sequential(*blocks)

########################################################################


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size=3,
                 dropRate=0.0, act=F.relu, norm='instance', stride=None):
        super(ResNetBasicBlock, self).__init__()
        self.norm = norm_dict[norm]()
        self.act = act
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kern_size, stride=1, bias=False,
                              padding='same')
        self.droprate = dropRate

    def _identity(self, x, target):
        to_pad = target.shape[1] - x.shape[1]
        return F.pad(x, pad=[0, 0, 0, to_pad, 0, 0])

    def forward(self, x):
        out = self.conv(self.act(self.norm(x)))
        if self.droprate > 0:
            out = F.dropout1d(out, p=self.droprate, training=self.training)
        return out + self._identity(x, out)


class ResNet(AlexNet):

    def _build_layers(self):
        blocks = []
        last_conv_chans = 1
        for conv_chan, ksize in zip(self.conv_channels, self.kern_sizes):
            blocks.append(ResNetBasicBlock(last_conv_chans, conv_chan, ksize,
                                           dropRate=self.droprate, norm=self.norm, stride=self.stride))
            last_conv_chans = conv_chan
        return nn.Sequential(*blocks)


#########################################################################

class DenseNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size=3,
                 dropRate=0.0, act=F.relu, norm='instance', stride=None):
        super(DenseNetBasicBlock, self).__init__()
        self.norm = norm_dict[norm]()
        self.act = act
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kern_size, stride=1,
                              padding="same", bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv(self.act(self.norm(x)))
        if self.droprate > 0:
            out = F.dropout1d(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class DenseNet(nn.Module):
    def __init__(self, conv_channels, kern_sizes, out_shape, inp_channels=1, act=F.gelu,
                 droprate=0.0, norm="instance", stride=None):
        super().__init__()
        self.conv_channels = conv_channels
        self.kern_sizes = kern_sizes
        self.inp_channels = inp_channels
        self.act = act
        self.norm = norm
        self.droprate = droprate
        self.conv_part = self._build_layers()
        self.onebyone_conv = nn.Conv1d(in_channels=sum(conv_channels) + inp_channels,
                                       out_channels=ONEBYONECHANNELS, kernel_size=1)
        self.logits = nn.LazyLinear(out_shape)

    def forward(self, x):
        conv_out = self.onebyone_conv(self.conv_part(x))
        flat_features = torch.flatten(self.act(conv_out), start_dim=1)
        if self.droprate > 0.0:
            flat_features = F.dropout(
                flat_features, self.droprate, training=self.training)
        return self.logits(flat_features)

    def _build_layers(self):
        blocks = []
        in_channels = self.inp_channels
        for conv_chan, ksize in zip(self.conv_channels, self.kern_sizes):
            blocks.append(DenseNetBasicBlock(in_channels, conv_chan, ksize,
                                             dropRate=self.droprate, norm=self.norm))
            in_channels += conv_chan
        return nn.Sequential(*blocks)


class SeqNet(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.seq_dim = 1

    def forward(self, x):
        preds = torch.stack([self.base(i) for i in torch.unbind(
            x, dim=self.seq_dim)], dim=self.seq_dim)
        return preds.mean(dim=self.seq_dim)


class DiscrimNet(nn.Module):
    def __init__(self, base_model, embed_dim=None, logits=5):
        super().__init__()
        self.base = base_model
        self.seq_dim = 1
        self.logits = nn.Sequential(
            nn.LazyLinear(32),
            nn.LeakyReLU(),
            nn.LazyLinear(logits)
        )
        self.discriminator = nn.Sequential(
            nn.LazyLinear(64),
            nn.LeakyReLU(),
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        preds = torch.stack([self.base(i) for i in torch.unbind(
            x, dim=self.seq_dim)], dim=self.seq_dim)
        scores = self.discriminator(preds)
        weighted_mean = (preds * scores).mean(self.seq_dim)
        # print(weighted_mean.shape)
        return self.logits(weighted_mean)


class SpecATNet(nn.Module):
    def __init__(self, base_model, embed_dim=8, logits=5):
        super().__init__()
        self.base = base_model
        self.att = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        self.logits = nn.LazyLinear(logits)
        self.seq_dim = 1
        self.last_att_weight = None

    def forward(self, x):
        preds = torch.stack([F.gelu(self.base(i)) for i in torch.unbind(
            x, dim=self.seq_dim)], dim=self.seq_dim)
        att, att_weight = self.att(preds, preds, preds)
        self.last_att_weight = att_weight
        return self.logits(att.mean(dim=self.seq_dim))


class TemplateNet(nn.Module):
    def __init__(self, base_model, templates, embed_dim=None, logits=5):
        super().__init__()
        self.base = base_model
        self.templates = torch.tensor(templates).to(device)
        self.seq_dim = 1

        self.discriminator = nn.Sequential(
            nn.LazyLinear(32),
            nn.LeakyReLU(),
            nn.LazyLinear(1)
        )

        self.logits = nn.Sequential(
            nn.LazyLinear(32),
            nn.LeakyReLU(),
            nn.LazyLinear(logits)
        )

    def forward(self, x):
        preds = torch.stack([self.base(i) for i in torch.unbind(
            x, dim=self.seq_dim)], dim=self.seq_dim)
        temp_preds = self.base(self.templates[:,None,:])
        similarities = (preds@temp_preds.T)  # batch, seqlen, n_templates
        scores = F.softmax(self.discriminator(similarities), dim=1)
        weighted_mean = (preds * scores).sum(self.seq_dim)
        return self.logits(weighted_mean)
