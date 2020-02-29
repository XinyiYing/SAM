import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *
from math import log10
from numpy import clip
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from vdsr_sam import Net_SAM
import numpy as np
from skimage import measure

# 使用单帧超分辨的数据集与训练模型来提升双目超分辨的性能

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=3, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=0.1")
parser.add_argument('--gamma', type=float, default=0.5, help='')
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to pretrained SISR model (default: none)")
parser.add_argument('--pretrained', default='../ckpt/VDSR/pretrain.pth', type=str, help='path to pretrained VDSR_SAM (default: none)')
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--scale", default=4, type=int, help="upscale factor (default: 4)")

torch.cuda.set_device(1)

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = TrainSetLoader('../data/train', opt.scale)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    # intervals can be one number: for every intervals a SAM
    # intervals also can be a list: for every layers_namber in list a SAM
    model = Net_SAM(n_intervals=[6,12], n_blocks=18, inchannels=1, nfeats=64, outchannels=1)

    print("===> Setting GPU")
    if opt.cuda:
        model = model.cuda()


    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # optionally train from a pretrained SISR
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            model_dict = model.state_dict()
            pretrained_dict = torch.load(opt.pretrained)['model'].state_dict()
            pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict1)
            model.load_state_dict(model_dict)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading model '{}'".format(opt.resume))
            weights = torch.load(opt.resume)
            opt.start_epoch = weights["epoch"] + 1
            model.load_state_dict(weights['model'])
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam([paras for paras in model.parameters() if paras.requires_grad == True], lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, epoch, scheduler)

def train(training_data_loader, optimizer, model, epoch, scheduler):
    scheduler.step()
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    loss_epoch = 0.0
    psnr_epoch = 0.0
    criterion_L1 = nn.L1Loss()
    criterion_mse = nn.MSELoss(size_average=False)

    for iteration, (HR_left, HR_right, LR_left, LR_right) in enumerate(training_data_loader):
        LR_left, LR_right, HR_left, HR_right = rgb2y(LR_left) / 255, rgb2y(LR_right) / 255, rgb2y(HR_left) / 255, rgb2y(HR_right) / 255
        b, c, h, w = LR_left.shape
        LR_left, LR_right, HR_left, HR_right = Variable(LR_left), Variable(LR_right), Variable(HR_left), Variable(HR_right)
        if opt.cuda:
            LR_left = LR_left.cuda()
            LR_right = LR_right.cuda()
            HR_left = HR_left.cuda()
            criterion_mse = criterion_mse.cuda()
            criterion_L1 = criterion_L1.cuda()

        HR_l, HR_r, map, mask = model(LR_left, LR_right)
        (M_right_to_left0, M_left_to_right0) = map[0]
        (M_right_to_left1, M_left_to_right1) = map[1]
        (V_right_to_left0, V_left_to_right0) = mask[0]
        (V_right_to_left1, V_left_to_right1) = mask[1]
        ###loss_SR

        loss_SR = criterion_mse(HR_l, HR_left)

        ### loss_photometric0
        LR_right_warped = torch.bmm(M_right_to_left0.contiguous().view(b * h, w, w),
                                    LR_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right0.contiguous().view(b * h, w, w),
                                   LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        loss_photo0 = criterion_L1(LR_left * V_left_to_right0, LR_right_warped * V_left_to_right0) + \
                     criterion_L1(LR_right * V_right_to_left0, LR_left_warped * V_right_to_left0)

        ### loss_photometric1
        LR_right_warped = torch.bmm(M_right_to_left1.contiguous().view(b * h, w, w),
                                    LR_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right1.contiguous().view(b * h, w, w),
                                   LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        loss_photo1 = criterion_L1(LR_left * V_left_to_right1, LR_right_warped * V_left_to_right1) + \
                      criterion_L1(LR_right * V_right_to_left1, LR_left_warped * V_right_to_left1)


        loss = loss_SR + 0.01 * (loss_photo0 + loss_photo1)

        loss_epoch = loss_epoch + loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(),opt.clip)
        optimizer.step()
        PSNR = cal_psnr(HR_left, HR_l)
        psnr_epoch = psnr_epoch + PSNR
    print("===> Epoch[{}]: Loss: {:.3f} PSNR: {:.3f} ".format(epoch, loss_epoch/(iteration+1), psnr_epoch/(iteration+1)))
    save_checkpoint_SAM(model, epoch)
    # valid('../data/valid', model)

def valid(Dataset,  model):
    test_set = TrainSetLoader(Dataset, opt.scale)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1,shuffle=True)
    psnr_epoch = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        LR_left, LR_right, HR_left = rgb2y(LR_left) / 255, rgb2y(LR_right) / 255, rgb2y(HR_left) / 255
        input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target = target.cuda()

        HR, _, _, _ = model(input_l, input_r)

        PSNR = cal_psnr(target, HR)
        psnr_epoch = psnr_epoch + PSNR
    print("===> Avg. PSNR: {:.8f} dB".format(psnr_epoch/(iteration+1)))

def save_checkpoint_SAM(model, epoch):
    model_out_path = "model_sam/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model.state_dict()}
    if not os.path.exists("model_sam/"):
        os.makedirs("model_sam/")
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

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

if __name__ == "__main__":
    main()