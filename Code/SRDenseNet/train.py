import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from SR_DenseNet_SAM import Net_SAM
from utils import *
from skimage import measure

# Training settings
parser = argparse.ArgumentParser(description="SR_DenseNet")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=30")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--pretrained', default='../ckpt/SRDenseNet/pretrain.pth', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate dacay')

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

    if opt.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
    model = Net_SAM(n_intervals=[3,6], growthRate=16, nDenselayer=8)
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    from functools import partial
    import pickle
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

	# optionally train form SISR network
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("===> load model {}".format(opt.pretrained))
            model_dict = model.state_dict()
            pretrained_dict = torch.load(opt.pretrained)['model'].state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("===> no model found at {}".format(opt.pretrained))

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
        LR_left, LR_right, HR_left, HR_right = rgb2y(LR_left) / 255, rgb2y(LR_right) / 255, rgb2y(HR_left) / 255, rgb2y(HR_right) / 255
        b, c, h, w = LR_left.shape
        input_l, input_r, target_l, target_r = Variable(LR_left), Variable(LR_right), Variable(HR_left), Variable(HR_right)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target_l = target_l.cuda()
        HR_l, HR_r, image_map, image_mask = model(input_l, input_r)

        ###loss_SR
        loss_SR  = criterion(HR_l, target_l)

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

        loss = loss_SR + 0.01 * (loss_photo)
        loss_epoch = loss_epoch + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        PSNR = cal_psnr(target_l, HR_l)
        psnr_epoch = psnr_epoch + PSNR
    print("===> Epoch[{}]: Loss: {:.3f} PSNR: {:.3f} ".format(epoch, loss_epoch / (iteration + 1),psnr_epoch / (iteration + 1)))
    # valid('../data/valid', model)

def valid(Dataset,  model):
    test_set = TrainSetLoader(Dataset)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1,shuffle=True)
    psnr_epoch = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        LR_left, LR_right, HR_left = rgb2y(LR_left) / 255, rgb2y(LR_right) / 255, rgb2y(HR_left) / 255
        input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target = target.cuda()
        HR, _, _ = model(input_l, input_r)

        PSNR = cal_psnr(target, HR)
        psnr_epoch = psnr_epoch + PSNR
    print("===> Avg. PSNR: {:.8f} dB".format(psnr_epoch/(iteration+1)))

def save_checkpoint_SAM(model, epoch):
	model_out_path = "model_SAM/" + "model_epoch_{}.pth".format(epoch)
	state = {"epoch": epoch, "model": model}
	# check path status
	if not os.path.exists("model/"):
		os.makedirs("model/")
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
