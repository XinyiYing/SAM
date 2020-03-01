import argparse
import torch
import numpy as np
import time, math
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import TestSetLoader, rgb2y
import matplotlib.pyplot as plt
from numpy import clip
from torchvision.transforms import ToPILImage
import scipy.io as scio
import os
from functools import partial
import pickle
from skimage import measure
from lapsrn import Net
from lapsrn_sam import Net_SAM

parser = argparse.ArgumentParser(description="Pytorch LapSRN Eval")
parser.add_argument("--model", type=str, default="../../ckpt/LapSRN/LapSRN.pth", help="model path")
parser.add_argument("--model_sam", type=str, default="../../ckpt/LapSRN/LapSRN_SAM1.pth", help="model path")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--scale", type=str, default=4, help="upscale factor")
parser.add_argument('--testset_dir', type=str, default='../data/test')
parser.add_argument('--dataset', type=str, default='middlebury')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

def valid_sam(test_loader, model):
    psnr_epoch = 0
    ssim_epoch = 0
    with torch.no_grad():
        for iteration, (HR_left, LR_left, LR_right, LR_left_x4) in enumerate(test_loader):
            LR_left_RGB, HR_left_RGB = LR_left_x4 / 255, HR_left / 255
            LR_left, LR_right, HR_left = rgb2y(LR_left) / 255, rgb2y(LR_right) / 255, rgb2y(HR_left) / 255
            input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left_RGB)
            if opt.cuda:
                input_l = input_l.cuda()
                input_r = input_r.cuda()
                target = target.cuda()
                LR_left_RGB = LR_left_RGB.cuda()
            _, HR, _, _, _, _ = model(input_l, input_r)
            HR = img_transfer(LR_left_RGB, HR)
            SR_left_np = np.array(torch.squeeze(HR[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
            HR_left_np = np.array(torch.squeeze(target[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
            PSNR = measure.compare_psnr(HR_left_np, SR_left_np)
            SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True)
            psnr_epoch = psnr_epoch + PSNR
            ssim_epoch = ssim_epoch +SSIM
        print("===> LapSRN_SAM Avg. PSNR: {:.8f} dB, Avg. SSIM: {:.8f}".format(psnr_epoch / (iteration + 1),
                                                                    ssim_epoch / (iteration + 1)))

def valid(test_loader, model):
    ssim_epoch = 0
    psnr_epoch = 0
    with torch.no_grad():
        for iteration, (HR_left, LR_left, _, LR_left_x4) in enumerate(test_loader):
            LR_left_rgb = LR_left_x4 / 255
            LR_left, HR_left = rgb2y(LR_left) / 255, HR_left / 255
            input_l, target_l = Variable(LR_left), Variable(HR_left)
            if opt.cuda:
                input_l = input_l.cuda()
                target_l = target_l.cuda()
                LR_left_rgb = LR_left_rgb.cuda()
            _, HR_l = model(input_l)
            HR_l = img_transfer(LR_left_rgb, HR_l)
            SR_left_np = np.array(torch.squeeze(HR_l[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
            HR_left_np = np.array(torch.squeeze(target_l[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
            PSNR = measure.compare_psnr(HR_left_np, SR_left_np)
            SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True)
            psnr_epoch = psnr_epoch + PSNR
            ssim_epoch = ssim_epoch + SSIM
        print("===> LapSRN Avg. PSNR: {:.8f} dB, Avg. SSIM: {:.8f}".format(psnr_epoch / (iteration + 1),
                                                                               ssim_epoch / (iteration + 1)))

def main():
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

    weights = torch.load(opt.model)
    model = Net()
    model.load_state_dict(weights['model'].state_dict())

    weights_sam = torch.load(opt.model_sam)
    model_sam = Net_SAM(n_intervals=[16,36], n_blocks=42, inchannels=1, nfeats=64, outchannels=1)
    model_sam.load_state_dict(weights_sam['model'].state_dict())
    model.cuda()
    model_sam.cuda()
    test_set = TestSetLoader(dataset_dir=opt.testset_dir + '/' + opt.dataset, scale_factor=opt.scale)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    import datetime
    oldtime = datetime.datetime.now()
    valid(test_loader, model)
    wotime = datetime.datetime.now()
    print('Time consuming: ', wotime - oldtime)
    valid_sam(test_loader, model_sam)
    wtime = datetime.datetime.now()
    print('Time consuming: ', wtime - wotime)

def img_transfer(img, img_y):
    img_r = img[:, 0, :, :]
    img_g = img[:, 1, :, :]
    img_b = img[:, 2, :, :]
    image_y = torch.squeeze(img_y, 1)
    image_cb = (-0.148 * img_r - 0.291 * img_g + 0.439 * img_b + 128 / 255.0)
    image_cr = (0.439 * img_r - 0.368 * img_g - 0.071 * img_b + 128 / 255.0)
    image_r = (1.164 * torch.unsqueeze((image_y - 16/255), 1) + 1.596 * torch.unsqueeze((image_cr - 128/255), 1))
    image_g = (1.164 * torch.unsqueeze((image_y - 16/255), 1) - 0.392 * torch.unsqueeze((image_cb - 128/255), 1) - 0.813 * torch.unsqueeze((image_cr - 128/255), 1))
    image_b = (1.164 * torch.unsqueeze((image_y - 16/255), 1) + 2.017 * torch.unsqueeze((image_cb - 128/255), 1))
    image = torch.cat((image_r, image_g, image_b), 1)
    return image

if __name__ == '__main__':
    main()
