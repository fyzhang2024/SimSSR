import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Net(nn.Module):
    def __init__(self, factor):
        super(Net, self).__init__()
        channels = 64
        n_block = 16
        self.factor = factor
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.cascade_resblocks = CascadeResBlocks(n_block, channels)
        self.up_sample = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(factor),
            nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, lf):
        b, u, v, c, h, w = lf.shape
        x = rearrange(lf, 'b u v c h w -> (b u v) c h w')
        buffer_init = self.init_conv(x)      #buffer_init.shape is (b u v) c h w
        buffer = rearrange(buffer_init, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        buffer = self.cascade_resblocks(buffer)
        buffer = rearrange(buffer, 'b u v c h w -> (b u v) c h w')
        beffer_final = buffer + buffer_init
        out = self.up_sample(beffer_final)
        out = rearrange(out, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)
        return out


class CascadeResBlocks(nn.Module):
    def __init__(self, n_block, channels):
        super(CascadeResBlocks, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(ResBlock(channels))
        self.Block = nn.Sequential(*Blocks)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return buffer

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        channel = channels
        channel_spa = channels
        channel_ang = channels
        channel_epi = channels
        self.spa_conv = nn.Sequential(
            nn.Conv2d(channel, channel_spa, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel_spa, channel, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.ang_conv = nn.Sequential(
            nn.Conv2d(channel, channel_ang, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel_ang, channel, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.epi_conv = nn.Sequential(
            nn.Conv2d(channel, channel_epi, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(channel_epi, channel, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, lf):
        b ,u ,v ,c ,h ,w = lf.shape
        x_spa = rearrange(lf, 'b u v c h w -> (b u v) c h w')
        buffer_spa = self.spa_conv(x_spa)
        buffer_spa = rearrange(buffer_spa, '(b u v) c h w -> b u v c h w', b=b, u=u, v=v)

        x_ang = rearrange(lf, 'b u v c h w -> (b h w) c u v')
        buffer_ang = self.ang_conv(x_ang)
        buffer_ang = rearrange(buffer_ang, '(b h w) c u v -> b u v c h w', b=b, h=h, w=w)

        x_epiH = rearrange(lf, 'b u v c h w -> (b u h) c v w')
        buffer_epiH = self.epi_conv(x_epiH)
        buffer_epiH = rearrange(buffer_epiH, '(b u h) c v w-> b u v c h w', b=b, u=u, h=h)

        x_epiV = rearrange(lf, 'b u v c h w -> (b v w) c u h')
        buffer_epiV = self.epi_conv(x_epiV)
        buffer_epiV = rearrange(buffer_epiV, '(b v w) c u h-> b u v c h w', b=b, v=v, w=w)

        return buffer_spa + buffer_ang + buffer_epiH + buffer_epiV + lf

if __name__ == "__main__":
    angRes = 5
    factor = 4
    net = Net(factor=factor).cuda()
    from thop import profile


    # 测整个Net
    input = torch.randn(1, angRes,  angRes, 1, 32, 32).cuda()
    flops, params = profile(net, inputs=(input,))
    print('=== Net整体 ===')
    print('   Number of parameters: %.2fK' % (params / 1e3))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))
    print()