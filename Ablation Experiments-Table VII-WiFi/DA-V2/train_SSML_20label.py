import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model_complexcnn_onlycnn import *
from VAT_DataAugmentation_20label_rotate2 import *
from center_loss import CenterLoss
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train(model, loss_center, train_dataloader, VAT_dataloader, optimizer, optimizer_cent, threshold, epoch, writer, device_num, batch_size):
    model.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    label_classifier_loss = 0
    unlabel_classifier_loss = 0
    cent_loss_x = 0
    cent_loss_u_w = 0
    cent_loss_u_s = 0
    result_loss = 0
    for num, (data_nnl, data_vat) in enumerate(zip(train_dataloader,VAT_dataloader)):
        data, target = data_nnl
        data_vat, target_vat = data_vat
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
            data_vat = data_vat.to(device)
        optimizer.zero_grad()
        optimizer_cent.zero_grad()

        inputs_x_w = data
        targets_x_w = target
        inputs_u_w = data_vat
        inputs_u_s, targets_u_s = StrongAugmentation(data_vat, target=torch.zeros((len(data_vat),)).cuda())

        inputs = torch.cat((inputs_x_w, inputs_u_w, inputs_u_s))
        targets_x = target

        output = model(inputs)
        logits = output[1]
        feature = output[0]

        feature_x = feature[:len(inputs_x_w)]
        feature_u_w = feature[len(inputs_x_w):len(inputs_x_w) + len(inputs_u_w)]
        feature_u_s = feature[len(inputs_x_w) + len(inputs_u_w):]

        logits_x = logits[:len(inputs_x_w)]
        logits_u_w = logits[len(inputs_x_w):len(inputs_x_w)+len(inputs_u_w)]
        logits_u_s = logits[len(inputs_x_w)+len(inputs_u_w):]


        Lx = F.cross_entropy(logits_x, targets_x_w, reduction='mean')

        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(threshold).float()
        we_can_believe = [index for index, value in enumerate(mask) if value == 1]

        targets_u = np.array(targets_u.cpu())
        mask = np.array(mask.cpu())
        targets_u_alls = np.zeros((1,))
        mask_alls = np.zeros((1,))
        for i in range(0, targets_u.shape[0]):
            targets_u_all = np.tile(targets_u[i], (1, 4))
            targets_u_all = targets_u_all.T
            targets_u_all = targets_u_all.reshape(-1)
            targets_u_all = targets_u_all.T

            mask_all = np.tile(mask[i], (1, 4))
            mask_all = mask_all.T
            mask_all = mask_all.reshape(-1)
            mask_all = mask_all.T

            targets_u_alls = np.concatenate([targets_u_alls, targets_u_all], axis=0)
            mask_alls = np.concatenate([mask_alls, mask_all], axis=0)
        targets_u_alls = torch.tensor(targets_u_alls[1:targets_u_alls.shape[0]]).long().cuda()
        mask_alls = torch.tensor(mask_alls[1:mask_alls.shape[0]]).float().cuda()

        Lu = (F.cross_entropy(logits_u_s, targets_u_alls, reduction='none') * mask_alls).mean()

        feature_x_norm = F.normalize(feature_x, p=2, dim=1)
        cent_loss_x_batch = loss_center(feature_x_norm, targets_x_w)[1]
        center_loss_u_w_batch = 0.0 * loss_center(feature_x_norm, targets_x_w)[1]
        if epoch >= 100:
            if len(we_can_believe) >= 2:
                feature_norm_u_w = F.normalize(feature_u_w, p=2, dim=1)
                center_loss_u_w_batch = loss_center(feature_norm_u_w[we_can_believe], torch.tensor(targets_u[we_can_believe]).long().cuda())[1]

        weight_cent = 0.001
        lambda_u = 1
        result_loss_batch = Lx + lambda_u * Lu + weight_cent * (cent_loss_x_batch + center_loss_u_w_batch)

        result_loss_batch.backward()
        optimizer.step()
        for param in loss_center.parameters():
            param.grad.data *= (1. / weight_cent)
        optimizer_cent.step()

        label_classifier_loss += Lx.item()
        unlabel_classifier_loss += Lu.item()
        cent_loss_x += cent_loss_x_batch.item()
        cent_loss_u_w += center_loss_u_w_batch.item()
        result_loss += result_loss_batch.item()

        classifier_output = F.log_softmax(logits_x, dim=1)
        pred = classifier_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    num = num + 1
    sample_number = num * batch_size
    label_classifier_loss /= num
    unlabel_classifier_loss /= num
    cent_loss_x /= num
    cent_loss_u_w /= len(train_dataloader)
    result_loss /= num

    print('Train Epoch: {} \tLabel_Classifier_Loss: {:.6f}, Unlabel_Classifier_Loss: {:.6f}, Label Center_Loss: {:.6f}, Unlabel Center_Loss: {:.6f}, Combined_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        label_classifier_loss,
        unlabel_classifier_loss,
        cent_loss_x,
        cent_loss_u_w,
        result_loss,
        correct,
        sample_number,
        100.0 * correct / sample_number)
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('LabelClassifierLoss/train', label_classifier_loss, epoch)
    writer.add_scalar('UnlabelClassifierLoss/train', unlabel_classifier_loss, epoch)

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

            inputs_x_w = data
            targets_x_w = target

            output = model(inputs_x_w)
            logits_x = output[1]

            test_loss += F.cross_entropy(logits_x, targets_x_w).item()

            pred = F.log_softmax(logits_x,dim=1).argmax(dim=1, keepdim=True)
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

def train_and_test(model, loss_center, train_dataset, VAT_dataset, val_dataset, optimizer, optimizer_cent, threshold, epochs, writer, save_path, device_num, batch_size, vat_batch_size):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        VAT_dataloader = DataLoader(VAT_dataset, batch_size=vat_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        train(model, loss_center, train_dataloader, VAT_dataloader, optimizer, optimizer_cent, threshold, epoch, writer, device_num, batch_size)
        test_loss = test(model, val_dataloader, epoch, writer, device_num)
        if test_loss <= current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved, the minimum loss is {}.".format(current_min_test_loss))
        print("------------------------------------------------")

class Config:
    def __init__(
        self,
        batch_size: int = 32,
        test_batch_size: int = 32,
        vat_batch_size: int = 100,
        epochs: int = 600,
        lr: float = 0.001,
        lr_cent: float = 0.001,
        n_classes: int = 16,
        save_path: str = 'model_weight/CNN_FixMatch_SSML_Norm_Rotate2_n_classes_16_20label_80unlabel_rand30.pth',
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
    writer = SummaryWriter("logs_FixMatch_SSML_Norm_Rotate2_20label_rand30")
    device = torch.device("cuda:"+str(conf.device_num))

    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)

    X_train_labeled, X_train_unlabeled, X_val, value_Y_train_labeled, value_Y_train_unlabeled, value_Y_val = TrainDataset(conf.ft, conf.rand_num)

    train_dataset = TensorDataset(torch.Tensor(X_train_labeled), torch.Tensor(value_Y_train_labeled))
    VAT_dataset = TensorDataset(torch.Tensor(X_train_unlabeled), torch.Tensor(value_Y_train_unlabeled))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))

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
                   train_dataset=train_dataset,
                   VAT_dataset=VAT_dataset,
                   val_dataset=val_dataset,
                   optimizer=optim,
                   optimizer_cent=optim_centloss,
                   threshold = conf.threshold,
                   epochs=conf.epochs,
                   writer=writer,
                   save_path=conf.save_path,
                   device_num=conf.device_num,
                   batch_size=conf.batch_size,
                   vat_batch_size=conf.vat_batch_size)

if __name__ == '__main__':
   main()