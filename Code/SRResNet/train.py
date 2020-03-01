import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet_sam import _NetG_SAM
from torchvision import models
import torch.utils.model_zoo as model_zoo
from utils import *
from functools import partial
import pickle
from skimage import measure

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--batchSize", type=int, default=5, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100,help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="../../ckpt/SRResNet/pretrain.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.5, help='Learning Rate decay')


def main():
    global opt, model, netContent
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
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True)

    if opt.vgg_loss:
        print('===> Loading VGG model')
        netVGG = models.vgg19()
        netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))

        class _content_model(nn.Module):
            def __init__(self):
                super(_content_model, self).__init__()
                self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])

            def forward(self, x):
                out = self.feature(x)
                return out

        netContent = _content_model()

    print("===> Building model")
    model = _NetG_SAM(n_intervals=[6,11], n_blocks=16, inchannels=3, nfeats=64, outchannels=3)
    criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        if opt.vgg_loss:
            netContent = netContent.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally train from SISR network
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch, scheduler)
        save_checkpoint_SAM(model, epoch)

def train(training_data_loader, optimizer, model, criterion, epoch, scheduler):

    scheduler.step()
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    criterion_L1 = L1Loss()
    loss_epoch = 0
    psnr_epoch = 0
    for iteration, (HR_left, HR_right, LR_left, LR_right) in enumerate(training_data_loader):
        b, c, h, w = LR_left.shape
        LR_left, LR_right, HR_left, HR_right = LR_left / 255, LR_right / 255, HR_left / 255, HR_right / 255
        input_l, input_r, target_l, target_r = Variable(LR_left), Variable(LR_right), Variable(HR_left), Variable(HR_right)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target_l = target_l.cuda()
        HR_l, HR_r, image_map, image_mask = model(input_l, input_r)

        #### loss_SR
        loss_SR = criterion(HR_l, target_l)

        ### loss_photometric
        length = len(image_map)
        loss_photo = 0
        for i in range(length):
            (M_right_to_left, M_left_to_right) = image_map[i]
            (V_right_to_left, V_left_to_right) = image_mask[i]
            LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                        input_r.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
            LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                                       input_l.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
            LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

            loss_photo = loss_photo + criterion_L1(input_l * V_left_to_right, LR_right_warped * V_left_to_right) + \
                         criterion_L1(input_r * V_right_to_left, LR_left_warped * V_right_to_left)

        loss = loss_SR + 0.01 * loss_photo

        if opt.vgg_loss:
            content_input = netContent(HR_l)
            content_target = netContent(target_l)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)

        optimizer.zero_grad()

        if opt.vgg_loss:
            netContent.zero_grad()
            content_loss.backward(retain_graph=True)

        loss_epoch = loss_epoch + loss
        if opt.vgg_loss:
            content_loss_epoch = content_loss_epoch + content_loss
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        PSNR = cal_psnr(target_l, HR_l)
        psnr_epoch = psnr_epoch + PSNR
    if opt.vgg_loss:
        print("===> Epoch[{}]: Loss: {:.5f} Content_loss {:.5f} PSNR: {:.5f}".format(epoch, loss_epoch / (iteration + 1),
                                                                                  content_loss_epoch / (iteration + 1),
                                                                                  psnr_epoch / (iteration + 1)))
    else:
        print("===> Epoch[{}]: Loss: {:.5f} PSNR: {:.5f}".format(epoch, loss_epoch / (iteration + 1),
                                                               psnr_epoch / (iteration + 1)))
    # valid('../../data/valid', model)

def valid(Dataset, model):
    test_set = TrainSetLoader(Dataset)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1,shuffle=True)
    psnr_epoch = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        LR_left, LR_right, HR_left = LR_left / 255, LR_right / 255, HR_left / 255
        input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target = target.cuda()

        HR, _, _ = model(input_l, input_r)

        PSNR = cal_psnr(target, HR)
        psnr_epoch = psnr_epoch + PSNR
    print("===> Avg. PSNR: {:.8f} dB".format(psnr_epoch / (iteration + 1)))

def save_checkpoint_SAM(model, epoch):
	model_out_path = "model_SAM/" + "model_epoch_{}.pth".format(epoch)
	state = {"epoch": epoch, "model": model}
	# check path status
	if not os.path.exists("model_SAM/"):
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
