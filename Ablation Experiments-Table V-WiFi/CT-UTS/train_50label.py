import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn_onlycnn import *
from VAT_DataAugmentation_50label_rotate2 import *
from center_loss import CenterLoss
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train(model, loss_center, train_dataloader, VAT_dataloader, optimizer, optimizer_cent, threshold, epoch, writer, device_num):
    model.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    label_classifier_loss = 0
    cent_loss_x = 0
    result_loss = 0
    for (data_nnl, data_vat) in zip(train_dataloader,VAT_dataloader):
        data, target = data_nnl
        data_vat, target_vat = data_vat
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
            data_vat = data_vat.to(device)
        optimizer.zero_grad()
        optimizer_cent.zero_grad()

        inputs_x_w, targets_x_w = WeakAugmentation(data, target)

        inputs = inputs_x_w
        targets_x = target

        output = model(inputs)
        logits = output[1]
        feature = output[0]

        feature_x = feature

        logits_x = logits

        Lx = F.cross_entropy(logits_x, targets_x_w, reduction='mean')

        feature_x_norm = F.normalize(feature_x, p=2, dim=1)
        cent_loss_x_batch = loss_center(feature_x_norm, targets_x_w)[1]

        weight_cent = 0.001
        result_loss_batch = Lx + weight_cent * cent_loss_x_batch

        result_loss_batch.backward()
        optimizer.step()
        for param in loss_center.parameters():
            param.grad.data *= (1. / weight_cent)
        optimizer_cent.step()

        label_classifier_loss += Lx.item()
        cent_loss_x += cent_loss_x_batch.item()
        result_loss += result_loss_batch.item()

        #判决的时候是集成判决
        logits_x_ensem = torch.zeros((int(logits_x.shape[0] / 4), int(logits_x.shape[1]))).cuda()
        for i in range(int(logits_x.shape[0] / 4)):
            logits_x_ensem[i] = logits_x[i * 4:(i + 1) * 4].mean(axis=0)

        classifier_output = F.log_softmax(logits_x_ensem, dim=1)
        pred = classifier_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    label_classifier_loss /= len(train_dataloader)
    cent_loss_x /= len(train_dataloader)
    result_loss /= len(train_dataloader)

    print('Train Epoch: {} \tLabel_Classifier_Loss: {:.6f}, Center_Loss: {:.6f}, Combined_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        label_classifier_loss,
        cent_loss_x,
        result_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('LabelClassifierLoss/train', label_classifier_loss, epoch)

def test(model, test_dataloader, epoch, writer, device_num):
    model.eval()
    test_loss = 0
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)

            inputs_x_w, targets_x_w = WeakAugmentation(data, target)

            output = model(inputs_x_w)
            logits_x = output[1]

            test_loss += F.cross_entropy(logits_x, targets_x_w).item()

            # 判决的时候是集成判决
            logits_x_ensem = torch.zeros((int(logits_x.shape[0] / 4), int(logits_x.shape[1]))).cuda()
            for i in range(int(logits_x.shape[0] / 4)):
                logits_x_ensem[i] = logits_x[i * 4:(i + 1) * 4].mean(axis=0)

            pred = F.log_softmax(logits_x_ensem,dim=1).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    writer.add_scalar('Accuracy/validation', 100.0 * correct / len(test_dataloader.dataset), epoch)
    writer.add_scalar('ClassifierLoss/validation', test_loss, epoch)

    return test_loss

def train_and_test(model, loss_center, train_dataloader, VAT_dataloader, val_dataloader, optimizer, optimizer_cent, threshold, epochs, writer, save_path, device_num):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_center, train_dataloader, VAT_dataloader, optimizer, optimizer_cent, threshold, epoch, writer, device_num)
        test_loss = test(model, val_dataloader, epoch, writer, device_num)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param, epoch)
        #     writer.add_histogram('{}.grad'.format(name), param.grad, epoch)

class Config:
    def __init__(
        self,
        batch_size: int = 32,
        test_batch_size: int = 32,
        vat_batch_size: int = 32,
        epochs: int = 300,
        lr: float = 0.001,
        lr_cent: float = 0.001,
        n_classes: int = 16,
        save_path: str = 'model_weight/CNN_FixMatch_Rotate2_n_classes_16_50label_50unlabel_rand30.pth',
        threshold: float = 0.95,
        device_num: int = 0,
        rand_num: int = 30,
        ft: int = 62,
        ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.vat_batch_size = vat_batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_cent = lr_cent
        self.n_classes = n_classes
        self.save_path = save_path
        self.threshold = threshold
        self.device_num = device_num
        self.rand_num = rand_num
        self.ft = ft

def main():
    conf = Config()
    writer = SummaryWriter("logs_FixMatch_Rotate2_50label_rand30")
    device = torch.device("cuda:"+str(conf.device_num))

    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train_labeled, X_train_unlabeled, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_val = TrainDataset(conf.ft, conf.rand_num)

    train_dataset = TensorDataset(torch.Tensor(X_train_labeled), torch.Tensor(value_Y_train_labeled))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)

    VAT_dataset = TensorDataset(torch.Tensor(X_train_unlabeled), torch.Tensor(value_Y_train_unlabeled))
    VAT_dataloader = DataLoader(VAT_dataset, batch_size=conf.vat_batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)

    model = base_complex_model()
    if torch.cuda.is_available():
        model = model.to(device)
    print(model)

    use_gpu = torch.cuda.is_available()
    loss_center = CenterLoss(num_classes=conf.n_classes, feat_dim=1024, use_gpu=use_gpu)


    optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0)
    optim_centloss = torch.optim.Adam(loss_center.parameters(), lr=conf.lr_cent, weight_decay=0)

    train_and_test(model,
                   loss_center = loss_center,
                   train_dataloader=train_dataloader,
                   VAT_dataloader=VAT_dataloader,
                   val_dataloader=val_dataloader,
                   optimizer=optim,
                   optimizer_cent=optim_centloss,
                   threshold = conf.threshold,
                   epochs=conf.epochs,
                   writer=writer,
                   save_path=conf.save_path,
                   device_num=conf.device_num)

if __name__ == '__main__':
   main()