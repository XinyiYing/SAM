import argparse
import torch
import numpy as np
import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import TestSetLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import os
from functools import partial
import pickle
from skimage import measure
from srresnet_sam import _NetG_SAM

parser = argparse.ArgumentParser(description="Pytorch SRResNet Eval")
parser.add_argument("--model_sam", type=str, default="./SRResNet_SAM.pth", help="model path")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--scale", type=str, default=4, help="upscale factor")
parser.add_argument('--testset_dir', type=str, default='../data/test')
parser.add_argument('--dataset_list', type=str, default=[ 'KITTI2012',  'KITTI2015', 'Middlebury'])
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


opt = parser.parse_args()
if opt.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

def valid_sam(testing_data_loader,  model):
    psnr_epoch = 0
    ssim_epoch = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        scene_name = testing_data_loader.dataset.file_list[iteration]
        LR_left, LR_right, HR_left = LR_left / 255, LR_right / 255, HR_left / 255
        input_l, input_r, HR = Variable(LR_left), Variable(LR_right), Variable(HR_left)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            HR = HR.cuda()
        SR_left, _, _, _ = model(input_l, input_r)
        SR_left = torch.clamp(SR_left, 0, 1)
        SR_left_np = np.array(torch.squeeze(SR_left[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
        HR_left_np = np.array(torch.squeeze(HR[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
        PSNR = measure.compare_psnr(HR_left_np, SR_left_np)
        SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True)
        psnr_epoch = psnr_epoch + PSNR
        ssim_epoch =ssim_epoch + SSIM
        ## save results
        if not os.path.exists('../Results/' + opt.dataset):
            os.makedirs('../Results/' + opt.dataset)
        SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left[:, :, :, 64:].data.cpu(), 0))
        SR_left_img.save('../Results/' + opt.dataset + '/' + scene_name + '.png')
    print("===> PSNR: {:.8f} dB SSIM: {:.8f} dB".format(psnr_epoch/(iteration+1), ssim_epoch/(iteration+1)))
    return psnr_epoch/(iteration+1), ssim_epoch/(iteration+1)

def main(opt):
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

    weights_sam = torch.load(opt.model_sam)
    model_sam = _NetG_SAM(n_intervals=[6,11])
    model_sam.load_state_dict(weights_sam['model'].state_dict())

    if opt.cuda:
        model_sam.cuda()
    test_set = TestSetLoader(dataset_dir=opt.testset_dir + '/' + opt.dataset, scale_factor=opt.scale)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    oldtime = datetime.datetime.now()
    psnr, ssim = valid_sam(test_loader, model_sam)
    newtime = datetime.datetime.now()
    print('Time consuming: ', newtime - oldtime)
    return psnr, ssim

if __name__ == '__main__':
    psnr_epoch = 0
    ssim_epoch = 0
    dataset_num = len(opt.dataset_list)
    for j in range(dataset_num):
        opt.dataset = opt.dataset_list[j]
        print(opt.dataset)
        psnr, ssim = main(opt)
        psnr_epoch = psnr_epoch + psnr
        ssim_epoch = ssim_epoch + ssim
    print("===> Avg. PSNR: {:.8f} dB SSIM: {:.8f} dB".format(psnr_epoch / len(opt.dataset_list),
                                                             ssim_epoch / len(opt.dataset_list)))

