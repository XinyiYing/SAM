import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np
import torch.nn.init as init


def xavier(param):
    init.xavier_uniform(param)


class SingleLayer(nn.Module):
    def __init__(self, inChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.conv = nn.Conv2d(inChannels, growthRate, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class Net_SAM(nn.Module):
    def __init__(self, n_intervals, growthRate=16, nDenselayer=8):
        super(Net_SAM, self).__init__()
        self.nintervals = n_intervals
        self.nbody = 8

        self.conv1 = nn.Conv2d(1, growthRate, kernel_size=3, padding=1, bias=True)
        inChannels = growthRate

        self.dense1 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer * growthRate

        self.dense2 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer * growthRate

        self.dense3 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer * growthRate

        self.dense4 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer * growthRate

        self.dense5 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer * growthRate

        self.dense6 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer * growthRate

        self.dense7 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer * growthRate

        self.dense8 = self._make_dense(inChannels, growthRate, nDenselayer)
        inChannels += nDenselayer * growthRate

        self.Bottleneck = nn.Conv2d(in_channels=inChannels, out_channels=256, kernel_size=1, padding=0, bias=True)

        self.convt1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1,
                                         bias=True)

        self.convt2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1,
                                         bias=True)

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1, bias=True)

        sam_layer = []
        for i in range(self.nbody):
            sam_layer.append(SAM(growthRate+nDenselayer * growthRate *(i+1)))
        self.sam_layer = nn.Sequential(*sam_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_dense(self, inChannels, growthRate, nDenselayer):
        layers = []
        for i in range(int(nDenselayer)):
            layers.append(SingleLayer(inChannels, growthRate))
            inChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, left, right):
        buffer_left, buffer_right = F.relu(self.conv1(left)), F.relu(self.conv1(right))
        image_map = []
        image_mask = []
        buffer_left, buffer_right = self.dense1(buffer_left), self.dense1(buffer_right)
        if 1 in self.nintervals:
            buffer_left, buffer_right, map = self.sam_layer[0](buffer_left, buffer_right)
            image_map.append(map)
        buffer_left, buffer_right = self.dense2(buffer_left), self.dense2(buffer_right)
        if 2 in self.nintervals:
            buffer_left, buffer_right, map = self.sam_layer[1](buffer_left, buffer_right)
            image_map.append(map)
        buffer_left, buffer_right = self.dense3(buffer_left), self.dense3(buffer_right)
        if 3 in self.nintervals:
            buffer_left, buffer_right, map, mask = self.sam_layer[2](buffer_left, buffer_right)
            image_map.append(map)
            image_mask.append(mask)
        buffer_left, buffer_right = self.dense4(buffer_left), self.dense4(buffer_right)
        if 4 in self.nintervals:
            buffer_left, buffer_right, map = self.sam_layer[3](buffer_left, buffer_right)
            image_map.append(map)
        buffer_left, buffer_right = self.dense5(buffer_left), self.dense5(buffer_right)
        if 5 in self.nintervals:
            buffer_left, buffer_right, map = self.sam_layer[4](buffer_left, buffer_right)
            image_map.append(map)
        buffer_left, buffer_right = self.dense6(buffer_left), self.dense6(buffer_right)
        if 6 in self.nintervals:
            buffer_left, buffer_right, map, mask = self.sam_layer[5](buffer_left, buffer_right)
            image_map.append(map)
            image_mask.append(mask)
        buffer_left, buffer_right = self.dense7(buffer_left), self.dense7(buffer_right)
        if 7 in self.nintervals:
            buffer_left, buffer_right, map = self.sam_layer[6](buffer_left, buffer_right)
            image_map.append(map)
        buffer_left, buffer_right = self.dense8(buffer_left), self.dense8(buffer_right)
        if 8 in self.nintervals:
            buffer_left, buffer_right, map = self.sam_layer[7](buffer_left, buffer_right)
            image_map.append(map)

        buffer_left, buffer_right = self.Bottleneck(buffer_left), self.Bottleneck(buffer_right)
        buffer_left, buffer_right = self.convt1(buffer_left), self.convt1(buffer_right)
        buffer_left, buffer_right = self.convt2(buffer_left), self.convt2(buffer_right)

        HR_l, HR_r = self.conv2(buffer_left), self.conv2(buffer_right)
        return HR_l, HR_r, image_map, image_mask

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
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
        self.bottleneck = nn.Conv2d(channels * 2 +1, channels, 1, 1, 0, bias=True)
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
        buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

        buffer_L = x_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

        out_L = self.bottleneck(torch.cat((buffer_l, x_left, V_left_to_right ), 1))
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
