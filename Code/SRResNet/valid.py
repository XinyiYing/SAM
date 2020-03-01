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
from srresnet import _NetG
from srresnet_sam import _NetG_SAM

parser = argparse.ArgumentParser(description="Pytorch SRResNet Eval")
parser.add_argument("--model", type=str, default="../../ckpt/SRResNet/SRResNet.pth", help="model path")
parser.add_argument("--model_sam", type=str, default="../../ckpt/SRResNet/SRResNet_SAM.pth", help="model path")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--scale", type=str, default=4, help="upscale factor")
parser.add_argument('--testset_dir', type=str, default='../../data/test')
parser.add_argument('--dataset', type=str, default='middlebury')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


opt = parser.parse_args()
if opt.cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

def valid_sam(testing_data_loader,  model):
    psnr_epoch = 0
    ssim_epoch = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        LR_left, LR_right, HR_left = LR_left / 255, LR_right / 255, HR_left / 255
        input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target = target.cuda()
        HR, _, _, _ = model(input_l, input_r)
        SR_left_np = np.array(torch.squeeze(HR[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
        HR_left_np = np.array(torch.squeeze(target[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
        PSNR = measure.compare_psnr(HR_left_np, SR_left_np)
        SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True)
        psnr_epoch = psnr_epoch + PSNR
        ssim_epoch =ssim_epoch + SSIM
    print("===> SRResNet_SAM Avg. PSNR: {:.8f} dB SSIM: {:.8f} dB".format(psnr_epoch/(iteration+1), ssim_epoch/(iteration+1)))

def valid(testing_data_loader, model):
    psnr_epoch = 0
    ssim_epoch = 0
    for iteration, (HR_left, _, LR_left, _) in enumerate(testing_data_loader):
        LR_left, HR_left = LR_left / 255, HR_left / 255
        input_l,  target_l = Variable(LR_left), Variable(HR_left)
        if opt.cuda:
            input_l = input_l.cuda()
            target_l = target_l.cuda()
        HR_l = model(input_l)
        SR_left_np = np.array(torch.squeeze(HR_l[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
        HR_left_np = np.array(torch.squeeze(target_l[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
        PSNR = measure.compare_psnr(HR_left_np, SR_left_np)
        SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True)
        psnr_epoch = psnr_epoch + PSNR
        ssim_epoch = ssim_epoch + SSIM
    print("===> SRResNet Avg. PSNR: {:.8f} dB SSIM: {:.8f} dB".format(psnr_epoch/(iteration+1), ssim_epoch/(iteration+1)))

def main():
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

    weights = torch.load(opt.model)
    model = _NetG()
    model.load_state_dict(weights['model'].state_dict())

    weights_sam = torch.load(opt.model_sam)
    model_sam = _NetG_SAM(n_intervals=[6,11])
    model_sam.load_state_dict(weights_sam['model'].state_dict())

    if opt.cuda:
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

def show(img):
    img = clip(img.data.cpu(), 0, 1)
    img = ToPILImage()(img[0,:,:,:])
    plt.figure(), plt.imshow(img)

def cal_psnr(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()
    return measure.compare_psnr(img1_np, img2_np)

if __name__ == '__main__':
    main()
