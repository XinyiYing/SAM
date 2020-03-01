import torch
import torch.nn as nn

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class _NetG_SAM(nn.Module):
    def __init__(self, n_intervals, n_blocks=16, inchannels=3, nfeats=64, outchannels=3):
        super(_NetG_SAM, self).__init__()
        self.n_blocks = n_blocks
        self.intervals = n_intervals
        if isinstance(n_intervals, list):
            self.nbody = len(n_intervals)
        if isinstance(n_intervals, int):
            self.nbody = self.n_blocks // n_intervals

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(64, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)

        sam_layer = []
        for _ in range(self.nbody):
            sam_layer.append(SAM(nfeats))
        self.sam_layer = nn.Sequential(*sam_layer)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, left, right):
        buffer_left, buffer_right = self.relu(self.conv_input(left)), self.relu(self.conv_input(right))
        residual_left, residual_right = buffer_left, buffer_right
        layers = 0
        image_map = []
        image_mask =[]
        for i in range(self.n_blocks):
            buffer_left, buffer_right = self.residual[i](buffer_left), self.residual[i](buffer_right)
            if isinstance(self.intervals, list):
                if (i + 1) in self.intervals:
                    buffer_left, buffer_right, map, mask = self.sam_layer[layers](buffer_left, buffer_right)
                    layers = layers + 1
                    image_map.append(map)
                    image_mask.append(mask)
            if isinstance(self.intervals, int):
                if (i + 1) % self.intervals == 0:
                    buffer_left, buffer_right, map, mask = self.sam_layer[layers](buffer_left, buffer_right)
                    layers = layers + 1
                    image_map.append(map)
                    image_mask.append(mask)
        buffer_left, buffer_right = self.bn_mid(self.conv_mid(buffer_left)), self.bn_mid(self.conv_mid(buffer_right))
        buffer_left, buffer_right = torch.add(buffer_left, residual_left), torch.add(buffer_right, residual_right)
        buffer_left, buffer_right = self.upscale4x(buffer_left), self.upscale4x(buffer_right)
        out_left, out_right = self.conv_output(buffer_left), self.conv_output(buffer_right)
        return out_left, out_right, image_map, image_mask


class _NetD(nn.Module):
    def __init__(self):
        super(_NetD, self).__init__()

        self.features = nn.Sequential(

            # input is (3) x 96 x 96
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 96 x 96
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 96 x 96
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (64) x 48 x 48
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 48 x 48
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 24 x 24
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (256) x 12 x 12
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (512) x 12 x 12
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, input):

        out = self.features(input)

        # state size. (512) x 6 x 6
        out = out.view(out.size(0), -1)

        # state size. (512 x 6 x 6)
        out = self.fc1(out)

        # state size. (1024)
        out = self.LeakyReLU(out)

        out = self.fc2(out)
        out = self.sigmoid(out)
        return out.view(-1, 1).squeeze(1)

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
        self.bottleneck = nn.Conv2d(channels * 2+1, channels, 1, 1, 0, bias=True)
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

        out_L = self.bottleneck(torch.cat((buffer_l, x_left, V_left_to_right), 1))
        out_R = self.bottleneck(torch.cat((buffer_r, x_right, V_right_to_left), 1))

        return out_L, out_R,\
               (M_right_to_left.contiguous().view(b, h, w, w),M_left_to_right.contiguous().view(b, h, w, w)),\
                (V_right_to_left, V_left_to_right)

def matric():
    net = _NetG_SAM([])
    from thop import profile
    flops, params = profile(net, (torch.ones(1, 3, 100, 100), torch.ones(1, 3, 100, 100)))
    print('   params: %.5fM' % (params / 1e6))
    print('   FLOPs: %.5fGFlops' % (flops/ 1e9))

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


