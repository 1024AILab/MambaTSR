# coding=utf-8
# @FileName:server_train.py
# @Time:2024/3/16 
# @Author: CZH
# coding=utf-8
# @FileName:train.py
# @Time:2024/3/14 
# @Author: CZH
import time

import torch.nn.functional
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torchvision import datasets
import numpy as np
import random
import csv
import logging
import torch
import torchvision.models as models
import torch.nn as nn
import os
from thop import profile
import torch
from torch.backends import cudnn
# import tqdm
import timm
# from timm.models.vision_transformer import vit_small_patch16_224, vit_base_patch8_224, vit_large_patch16_224

# from models_mamba import vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2

from demomain import parse_option, main_demo


def dataloader_prepare(train_data_folder_path, test_data_folder_path, batchsize):
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize([32, 32]),
    ])

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=(0.8), contrast=(1, 1)),
        transforms.Resize([32, 32]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

    ])

    train_data = datasets.ImageFolder(train_data_folder_path, transform=transform_train)
    test_data = datasets.ImageFolder(test_data_folder_path, transform=transform_val)

    # 数据集的长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练数据集的长度为：{}".format(train_data_size))
    print("测试数据集的长度为：{}".format(test_data_size))

    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    # bs的值可以随意调节，网络最后的输出必须是[bs,10]十个数
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=True)

    return train_dataloader, test_dataloader


# # model=lightViT(img_size=(32,32),depth=3,patch_size=8,mlp_ratio=2,num_heads=6,embed_dim=192,num_classes=43)
# model = lightViT(img_size=(32, 32), depth=5, patch_size=8, mlp_ratio=2, num_heads=6, embed_dim=192, num_classes=43)
# model = model.to(device)
# # model=Module2()

# -----------------------------------train /test

def train_and_test(start_epoch, num_epochs, train_dataloader, test_dataloader, model,
                   device, batchsize, class_num, model_name, dataset_name, path, test_data_path):
    folder_path = os.path.join(path, model_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("{}文件夹已创建".format(folder_path))

    folder_path = os.path.join(folder_path, dataset_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("{}文件夹已创建".format(folder_path))

    log_name = 'training.log'
    training_log = os.path.join(folder_path, log_name)
    logging.basicConfig(filename=training_log, level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info(f"Start Training!!!")
    logging.info(f"using the model {model_name} to training")
    save_check_point_name = model_name + "_train_on_" + dataset_name + '.pth'
    save_check_point_name = os.path.join(folder_path, save_check_point_name)
    best_acc = 0
    best_acc_epoch = 0
    warmup_epochs = 5
    init_lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay=5e-3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=[30, 40],
                                                     gamma=0.1, last_epoch=-1)
    model = model.to(device)
    # 定义一个空列表用于存储损失值
    losses_saver = []
    # scaler = torch.cuda.amp.GradScaler() #
    for epoch in range(start_epoch, num_epochs):
        model.train()
        losses = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data = data.to(device)
                target = target.to(device)
                # with torch.cuda.amp.autocast():
                preds = model(data)
                # print(preds.shape)[96,10]
                loss = criterion(preds, target)
                optimizer.zero_grad()
                # scaler.scale(loss).backward() #
                loss.backward()
                optimizer.step()
                # scaler.step(optimizer) #
                # scaler.update() #
                losses += loss
                tepoch.set_postfix(loss='{:.3f}'.format(round((losses.item() / batchsize * 1.0), 4)))
            losses_saver.append(round(losses.item() * 1.0, 4))
        """
            设置到预测模式，并且no_grad()关闭梯度计算
            predictions=preds.max(1).indices：每行最大值的索引
            """

        num_correct = 0
        num_samples = 0

        # 在训练循环之前初始化这两个数组
        num_classes = class_num
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))

        # 在测试循环中，使用真实类别标签来索引这两个数组，更新正确预测和总数
        with torch.no_grad():
            model.eval()
            for x, y in (test_dataloader):
                x = x.to(device)
                y = y.to(device)
                preds = model(x)
                predictions = preds.max(1).indices
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                _, predicted = torch.max(preds, 1)
                c = (predicted == y).squeeze()
                for i in range(len(y)):
                    label = y[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
            acc = (num_correct / num_samples).item()

            print('Accuracy:{:4f}'.format(acc), end=' ')
            # print('\t last_lr:', scheduler.get_last_lr()[0])
            print(optimizer.state_dict()['param_groups'][0]['lr'])
            # 在测试循环结束后，计算并打印每个类别的正确率
            # for i in range(num_classes):
            #     print('Accuracy of %5s : %2d %%' % (
            #         i, 100 * class_correct[i] / class_total[i]))
            logging.info(
                f"Epoch {epoch}/{num_epochs}  Accuracy:{round(acc, 4)} Loss:{round((losses.item() / batchsize * 1.0), 4)}")
        scheduler.step()
        model.train()

        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch
            # save it 在循环内
            checkpoint_dict = {'epoch': epoch,
                               'model_state_dict': model.state_dict(),
                               'optim_state_dict': optimizer.state_dict(),
                               'criterion_state_dict': criterion.state_dict()}
            torch.save(checkpoint_dict, save_check_point_name)
        logging.info(f"Best Accuracy:{best_acc} at {round(best_acc_epoch, 4)} epoch")

    csv_file = 'losses.csv'
    csv_file = model_name + csv_file
    csv_file = os.path.join(folder_path, csv_file)
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Loss'])
        for loss in losses_saver:
            writer.writerow([loss])
    print(f"损失值已保存到 {csv_file}")

    input = torch.zeros((1, 3, 32, 32)).to(device)
    flops, params = profile(model, inputs=(input,))

    print("\n参数量：", round(params / 1.0e6, 4), "M")
    print("FLOPS：", round(flops / 1.0e6, 4), "M")
    information_density = params / flops
    print("信息密度：", information_density)
    logging.info(
        f"参数量: {round(params / 1.0e6, 4)}M, FLOPS: {round(flops / 1.0e6, 4)}M, 信息密度:{information_density}")
    average_time, fps = get_fps_new(model=model, 
                                    test_data_folder_path=test_data_path, 
                                    batch_size=batchsize)
    logging.info(
        f"平均推理时间: {round(average_time, 4)} ms, FPS: {round(fps, 4)}")


def get_fps_new(model, test_data_folder_path, batch_size):

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize([32, 32]),
    ])

    test_data = datasets.ImageFolder(test_data_folder_path, transform=transform_val)

    # 数据集的长度
    test_data_size = len(test_data)
    print("测试数据集的长度为：{}".format(test_data_size))

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        for inputs, _ in test_loader:
            outputs = model(inputs)
        end_time = time.time()

    # 4. 分析结果
    inference_time = end_time - start_time
    inference_time / len(test_data) * 1000
    fps = 1 / (inference_time / len(test_data))
    print(f"Inference time for one batch: {inference_time / len(test_data) * 1000} ms")
    print(f"Frames per second (FPS): {fps}")
    return inference_time, fps


def get_args():
    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(description='这是一个简单的示例程序')
    # 添加命令行参数
    # german 43
    # china 103
    # india 15
    parser.add_argument('--dataset_name', default="GTSRB_128x128",
                        type=str, help='输出文件路径')

    parser.add_argument('--class_num', default=43,
                        type=int, help='输入文件路径')

    parser.add_argument('--batch_size', default=64,
                        type=int, help='是否显示详细信息')
    # autodl-tmp/GTSRB_128x128/train
    parser.add_argument('--train_dataset_path',
                        default=r"/root/autodl-tmp/GTSRB_128x128/train",
                        type=str, help='是否显示详细信息')
    # /root/autodl-tmp/GTSRB_128x128/train
    parser.add_argument('--test_dataset_path',
                        default=r"/root/autodl-tmp/GTSRB_128x128/test",
                        type=str, help='是否显示详细信息')

    parser.add_argument('--start_epoch', default=0,
                        type=int, help='是否显示详细信息')

    parser.add_argument('--num_epoch', default=100,
                        type=int, help='是否显示详细信息')
    # 解析命令行参数
    args = parser.parse_args()

    # 使用解析后的参数
    # input_file = args.input
    # output_file = args.output
    # verbose = args.verbose

    return args


class ModifiedVGG16(nn.Module):
    def __init__(self, num_classes=43):
        super(ModifiedVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=False)
        num_features = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)


class Modified_resnet18(nn.Module):
    def __init__(self, num_classes=43):
        super(Modified_resnet18, self).__init__()
        self.vgg16 = models.VisionTransformer.resnet18(num_classes=num_classes, pretrained=False)
        # num_features = self.vgg16.classifier[-1].in_features
        # self.vgg16.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)


if __name__ == '__main__':
    """
    混合精度？尝试提速
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    args = get_args()
    train_dataloader, test_dataloader = dataloader_prepare(args.train_dataset_path,
                                                           args.test_dataset_path,
                                                           args.batch_size)

    # model = ModifiedVGG16(num_classes=args.class_num)
    # model = vit_small_patch16_224(img_size=(32, 32), depth=5, patch_size=8, mlp_ratio=2,
    #                               num_heads=6, embed_dim=192, num_classes=args.class_num)
    # args1 = get_args_p
    _, config = parse_option()
    model = main_demo(config)

    model_name = "swin vim"
    path = r"/root"
    train_and_test(args.start_epoch,
                   args.num_epoch,
                   train_dataloader,
                   test_dataloader,
                   model, device,
                   args.batch_size,
                   args.class_num, model_name,
                   args.dataset_name,
                   path=path, test_data_path=args.test_dataset_path)
    # get_fps(model)



