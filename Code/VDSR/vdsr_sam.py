import torch
import torch.nn as nn
from math import sqrt

class Net_SAM(nn.Module):
    def __init__(self, n_intervals, n_blocks=18, inchannels = 1, nfeats =64, outchannels = 1):
        super(Net_SAM, self).__init__()
        self.n_blocks = n_blocks
        self.intervals = n_intervals
        if isinstance(n_intervals, list):
            self.nbody = len(n_intervals)
        if isinstance(n_intervals, int):
            self.nbody = self.n_blocks // n_intervals

        self.residual_layer = self.make_layer(Conv_ReLU_Block, self.n_blocks)
        self.input = nn.Conv2d(inchannels, nfeats, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(nfeats, outchannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        sam_layer = []
        for _ in range(self.nbody):
            sam_layer.append(SAM(nfeats))
        self.sam_layer = nn.Sequential(*sam_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


    def forward(self, left, right):
        buffer_left, buffer_right = self.relu(self.input(left)), self.relu(self.input(right))
        layers = 0
        image_map = []
        image_mask = []
        for i in range(self.n_blocks):
            buffer_left, buffer_right = self.residual_layer[i](buffer_left), self.residual_layer[i](buffer_right)
            if isinstance(self.intervals, list):
                if (i+1) in self.intervals:
                    buffer_left, buffer_right, \
                    map, mask = self.sam_layer[layers](buffer_left, buffer_right)
                    image_map.append(map)
                    image_mask.append(mask)
                    layers = layers + 1
            if isinstance(self.intervals, int):
                if (i+1) % self.intervals == 0:
                    buffer_left, buffer_right, \
                    map = self.sam_layer[layers](buffer_left, buffer_right)
                    map, mask = self.sam_layer[layers](buffer_left, buffer_right)
                    image_map.append(map)
                    image_mask.append(mask)
                    layers = layers + 1
        buffer_left, buffer_right = self.output(buffer_left), self.output(buffer_right)
        out_left, out_right = buffer_left + left, buffer_right + right

        return out_left, out_right, image_map, image_mask

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.conv(x))
        return out

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class SAM(nn.Module):# stereo attention block
    def __init__(self, channels):
        super(SAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.rb = ResB(64)
        self.softmax = nn.Softmax(-1)
        self.bottleneck = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)
    def forward(self, x_left, x_right):# B * C * H * W
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)
        ### M_{right_to_left
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

        return out_L, out_R, \
               (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
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

def metric():
    net = Net_SAM([6,12])
    from thop import profile
    flops, params = profile(net, (torch.ones(1, 1, 100, 100), torch.ones(1, 1, 100, 100)))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   params: %.5fM' % (total / 1e6))
    print('   FLOPs: %.5fGFlops' % (flops/ 1e9))