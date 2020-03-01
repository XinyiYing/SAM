import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        output = self.cov_block(x)
        return output


class Net_SAM(nn.Module):
    def __init__(self, n_intervals, n_blocks=42, inchannels = 1, nfeats =64, outchannels = 1):
        super(Net_SAM, self).__init__()
        self.n_blocks = n_blocks
        self.intervals = n_intervals
        if isinstance(n_intervals, list):
            self.nbody = len(n_intervals)
        if isinstance(n_intervals, int):
            self.nbody = self.n_blocks // n_intervals

        self.conv_input = nn.Conv2d(inchannels, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.convt_I1 = nn.ConvTranspose2d(inchannels, inchannels, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R1 = nn.Conv2d(nfeats, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)

        self.convt_I2 = nn.ConvTranspose2d(inchannels, inchannels, kernel_size=4, stride=2, padding=1,
                                           bias=False)
        self.convt_R2 = nn.Conv2d(nfeats, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)

        sam_layer = []
        for _ in range(self.nbody):
            sam_layer.append(SAM(nfeats))
        self.sam_layer = nn.Sequential(*sam_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, left, right):
        buffer_left, buffer_right = self.relu(self.conv_input(left)), self.relu(self.conv_input(right))
        # convt_F1------------------------------------------------------------------------------------------------------
        layers = 0
        image_map = []
        image_mask = []
        for i in range(int(self.n_blocks/2)):
            buffer_left, buffer_right = self.convt_F1[0].cov_block[i](buffer_left), self.convt_F1[0].cov_block[i](buffer_right)
            if isinstance(self.intervals, list):
                if (i + 1) in self.intervals:
                    buffer_left, buffer_right, map, mask = self.sam_layer[layers](buffer_left, buffer_right)
                    image_map.append(map)
                    image_mask.append(mask)
                    layers = layers + 1
            if isinstance(self.intervals, int):
                if (i + 1) % self.intervals == 0:
                    buffer_left, buffer_right, map, mask = self.sam_layer[layers](buffer_left, buffer_right)
                    image_map.append(map)
                    image_mask.append(mask)
                    layers = layers + 1
        # --------------------------------------------------------------------------------------------------------------
        convt_l_I1, convt_r_I1 = self.convt_I1(left), self.convt_I1(right) # 原图上采样
        convt_l_R1, convt_r_R1 = self.convt_R1(buffer_left), self.convt_R1(buffer_right) # 64->1
        HR_l_2x, HR_r_2x = convt_l_I1 + convt_l_R1, convt_r_I1 + convt_r_R1

        # convt_F2------------------------------------------------------------------------------------------------------
        for i in range(int(self.n_blocks/2)):
            buffer_left, buffer_right = self.convt_F2[0].cov_block[i](buffer_left), self.convt_F2[0].cov_block[i](buffer_right)
            if isinstance(self.intervals, list):
                if (i + 1 + int(self.n_blocks/2)) in self.intervals:
                    buffer_left, buffer_right, map, mask = self.sam_layer[layers](buffer_left, buffer_right)
                    image_map.append(map)
                    image_mask.append(mask)
                    layers = layers + 1
            if isinstance(self.intervals, int):
                if (i + 1 + int(self.n_blocks/2)) % self.intervals == 0:
                    buffer_left, buffer_right, map, mask = self.sam_layer[layers](buffer_left, buffer_right)
                    image_map.append(map)
                    image_mask.append(mask)
                    layers = layers + 1
        # --------------------------------------------------------------------------------------------------------
        convt_l_I2, convt_r_I2 = self.convt_I2(HR_l_2x), self.convt_I2(HR_r_2x)  # 原图上采样
        convt_l_R2, convt_r_R2 = self.convt_R2(buffer_left), self.convt_R2(buffer_right)  # 64->1
        HR_l_4x, HR_r_4x = convt_l_I2 + convt_l_R2, convt_r_I2 + convt_r_R2

        return HR_l_2x, HR_l_4x, HR_r_2x, HR_r_4x, image_map, image_mask


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        )


    def forward(self, x):
        out = self.body(x)
        return out + x

class SAM(nn.Module):# stereo attention block
    def __init__(self, channels):
        super(SAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.rb = RB(channels)
        self.softmax = nn.Softmax(-1)
        self.bottleneck = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)
    def forward(self, x_left, x_right):# B * C * H * W
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)
        ### M_{right_to_left}
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)  # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)  # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))  # (B*H) * W * W
        M_right_to_left = self.softmax(score)

        score_T = score.permute(0,2,1)
        M_left_to_right = self.softmax(score_T)

        # valid mask
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)
        V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
        V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
        V_right_to_left = morphologic_process(V_right_to_left)

        buffer_R = x_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1,
                                                                                              2)  # B * C * H * W

        buffer_L = x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1,
                                                                                              2)  # B * C * H * W

        out_L = self.bottleneck(torch.cat((buffer_l, x_left, V_left_to_right), 1))
        out_R = self.bottleneck(torch.cat((buffer_r, x_right, V_right_to_left), 1))

        return out_L, out_R,\
               (M_right_to_left.contiguous().view(b, h, w, w),M_left_to_right.contiguous().view(b, h, w, w)),\
                 (V_right_to_left, V_left_to_right)



import numpy as np
from skimage import morphology
def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)


    return torch.from_numpy(mask_np).float().to(device)


def structure_metric():
    net = Net_SAM([36])
    from thop import profile
    flops, params = profile(net, (torch.ones(1, 1, 100, 100), torch.ones(1, 1, 100, 100)))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   params: %.5fM' % (total / 1e6))
    print('   FLOPs: %.5fGFlops' % (flops/ 1e9))
    sam = SAM(64)
    flops, params = profile(sam, (torch.ones(1, 64, 128, 128), torch.ones(1, 64, 128, 128)))
    print(flops/1e9)
    print(params)