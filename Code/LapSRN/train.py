import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from lapsrn import Net, L1_Charbonnier_loss
from lapsrn_sam import Net_SAM, L1_Charbonnier_loss
from utils import *
from skimage import measure


# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--batchSize", type=int, default=12, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--pretrained", default="../../ckpt/LapSRN/pretrain.pth", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def main():

    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = TrainSetLoader('../../data/train')
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    # intervals can be one number: for every intervals a SAM(require: intervals % 2 == 0 )
    # intervals also can be a list: for every layers_namber in list a SAM(require: (for num in intervals) % 2 == 0 )
    model = Net_SAM(n_intervals=[16,36], n_blocks=42, inchannels=1, nfeats=64, outchannels=1)
    criterion = L1_Charbonnier_loss()

    print("===> Setting GPU")
    if opt.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            weights = torch.load(opt.resume)
            opt.start_epoch = weights["epoch"] + 1
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally train from SISR network
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            model_dict = model.state_dict()
            pretrained_dict = torch.load(opt.pretrained)['model'].state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    criterion_L1 = L1Loss()
    loss_epoch = 0.0
    psnr_epoch = 0.0
    for iteration, (HR_left, HR_right, LR_left, LR_right, LR2_left, LR2_right) in enumerate(training_data_loader):
        LR_left, LR_right = rgb2y(LR_left) / 255, rgb2y(LR_right) / 255
        b, c, h, w = LR_left.shape
        HR_left, HR_right = rgb2y(HR_left) / 255, rgb2y(HR_right) / 255
        LR2_left, LR2_right = rgb2y(LR2_left) / 255, rgb2y(LR2_right) / 255
        input_l, input_r, HR_left, HR_right = Variable(LR_left), Variable(LR_right), Variable(HR_left), Variable(HR_right)
        LR2_left, LR2_right = Variable(LR2_left), Variable(LR2_right)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            HR_left = HR_left.cuda()
            LR2_left = LR2_left.cuda()
            LR2_right = LR2_right.cuda()

        HR_2x_left, HR_4x_left, HR_2x_right, HR_4x_right, image_map, image_mask = model(input_l,input_r)

        loss_x2_left = criterion(HR_2x_left, LR2_left)
        loss_x4_left = criterion(HR_4x_left, HR_left)

        loss_SR = loss_x2_left + loss_x4_left

        ### loss_photometric
        loss_photo = 0
        (M_right_to_left, M_left_to_right) = image_map[0]
        (V_right_to_left0, V_left_to_right0) = image_mask[0]
        LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                    input_r.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                                   input_l.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)


        loss_photo = loss_photo + criterion_L1(input_l* V_left_to_right0, LR_right_warped* V_left_to_right0) + \
                     criterion_L1(input_r * V_right_to_left0, LR_left_warped * V_right_to_left0)

        (M_right_to_left, M_left_to_right) = image_map[1]
        (V_right_to_left0, V_left_to_right0) = image_mask[1]
        LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b * h*2, w*2, w*2),
                                    LR2_right.permute(0, 2, 3, 1).contiguous().view(b * h*2, w*2, c))
        LR_right_warped = LR_right_warped.view(b, h*2, w*2, c).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h*2, w*2, w*2),
                                   LR2_left.permute(0, 2, 3, 1).contiguous().view(b * h*2, w*2, c))
        LR_left_warped = LR_left_warped.view(b, h*2, w*2, c).contiguous().permute(0, 3, 1, 2)


        loss_photo = loss_photo + criterion_L1(LR2_left* V_left_to_right0, LR_right_warped* V_left_to_right0) + \
                     criterion_L1(LR2_right * V_right_to_left0, LR_left_warped * V_right_to_left0)

        loss = loss_SR + 0.01 * (loss_photo)
        loss_epoch = loss_epoch + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        PSNR = cal_psnr(HR_left, HR_4x_left)
        psnr_epoch = psnr_epoch + PSNR
    print("===> Epoch[{}]: Loss: {:.3f} PSNR: {:.3f} ".format(epoch, loss_epoch/(iteration+1), psnr_epoch/(iteration+1)))
    save_checkpoint_SAM(model, epoch)
    # valid('../../data/valid', model)

def valid(Dataset,  model):
    test_set = ValSetLoader(Dataset)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1,shuffle=True)
    psnr_epoch = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        LR_left, LR_right, HR_left = rgb2y(LR_left) / 255, rgb2y(LR_right) / 255, rgb2y(HR_left) / 255
        input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target = target.cuda()

        _, HR_l_4x, _, _, _ = model(input_l, input_r)

        PSNR = cal_psnr(target, HR_l_4x)
        psnr_epoch = psnr_epoch + PSNR
    print("===> Avg. PSNR: {:.8f} dB".format(psnr_epoch/(iteration+1)))

def save_checkpoint_SAM(model, epoch):
	model_out_path = "model_SAM/" + "model_epoch_{}.pth".format(epoch)
	state = {"epoch": epoch, "model": model}
	# check path status
	if not os.path.exists("model_SAM"):
		os.makedirs("model_SAM/")
	# save model
	torch.save(state, model_out_path)
	print("Checkpoint saved to {}".format(model_out_path))

def cal_psnr(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()
    return measure.compare_psnr(img1_np, img2_np)

if __name__ == "__main__":
    main()
