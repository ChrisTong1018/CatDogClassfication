from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

data_path = 'dogs-vs-cats'
os.makedirs('result', exist_ok=True)

# 加载数据
class CustomDataset(Dataset):
    # 初始化数据集，接收数据路径和数据增强变换
    def __init__(self, data_path, is_train=True, transform=None):
        self.data = data_path
        self.is_train = is_train
        self.transform = transform

    # 返回数据集大小
    def __len__(self):
        return len(self.data)
    
    # 获取指定索引的数据样本，加载图像并根据文件名确定标签，应用数据增强变换后返回图像和标签
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.is_train:
            img_path = os.path.join(data_path, 'training_set', sample)
        else:
            img_path = os.path.join(data_path, 'test_set', sample)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Image not found: {img_path}')
        img = Image.open(img_path).convert('RGB')   # 加载图像并转换为RGB格式
        label = 0 if 'cat' in sample else 1
        if self.transform:
            img = self.transform(img)
        return img, label
    
# 高斯噪声
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'
    
# 定义数据增强流程
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # 统一调整图像大小为256x256
    transforms.RandomRotation(degrees=15),   # 随机旋转
    transforms.RandomHorizontalFlip(p=0.5),   # 随即水平翻转
    transforms.ColorJitter(brightness=0.4, contrast=0.4),   # 增加亮度、对比度
    transforms.CenterCrop(224),   # 中心裁剪为目标大小
    transforms.ToTensor(),   # 转换为Tensor
    AddGaussianNoise(mean=0.0, std=0.1)   # 添加高斯噪声
])

# 创建训练集和测试集
train_data = [i for i in os.listdir(os.path.join(data_path, 'training_set')) if i.endswith('.jpg')]
train_dataset = CustomDataset(data_path=train_data, is_train=True, transform=transform)
test_data = [i for i in os.listdir(os.path.join(data_path, 'test_set')) if i.endswith('.jpg')]
test_dataset = CustomDataset(data_path=test_data, is_train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层部分，包含三个卷积层，每个卷积层后面跟着批归一化、最大池化和ReLU激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2),   # 输入通道数3，输出通道数8，卷积核大小3x3，步长2
            nn.BatchNorm2d(8),    # 批归一化层，规范化卷积层输出
            nn.MaxPool2d(2, 2),    # 最大池化层，池化窗口大小2x2，步长2
            nn.ReLU(),   # 激活函数，增加非线性
            nn.Conv2d(8, 16, kernel_size=3, stride=2),   # 输入通道数8，输出通道数16，卷积核大小3x3，步长2
            nn.BatchNorm2d(16),    # 批归一化层，规范化卷积层输出
            nn.MaxPool2d(2, 2),    # 最大池化层，池化窗口大小2x2，步长2
            nn.ReLU(),   # 激活函数，增加非线性
            nn.Conv2d(16, 32, kernel_size=3, stride=2),   # 输入通道数16，输出通道数32，卷积核大小3x3，步长2
            nn.BatchNorm2d(32),    # 批归一化层，规范化卷积层输出
            nn.MaxPool2d(2, 2),    # 最大池化层，池化窗口大小2x2，步长2
            nn.ReLU(),   # 激活函数，增加非线性
        )

        # 全连接层部分，包含一个全连接层，后面跟着批归一化、ReLU激活函数和Dropout层，最后输出两类的预测结果
        self.fc = nn.Sequential(
            nn.Flatten(),    # 将卷积层输出展平为一维向量
            nn.Linear(288, 128),    # 全连接层，输入特征数288，输出特征数128
            nn.BatchNorm1d(128),   # 批归一化层，规范化全连接层输出
            nn.ReLU(),   # 激活函数，增加非线性
            nn.Dropout(0.5),   # Dropout层，随机丢弃50%的神经元，防止过拟合
            nn.Linear(128, 2)   # 全连接层，输入特征数128，输出特征数2（猫和狗两类）
        )

    # 前向传播函数，定义数据通过网络的流程，先通过卷积层部分提取特征，再通过全连接层部分进行分类预测
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CNN().to(device)
criterion = nn.CrossEntropyLoss()   # 定义损失函数，使用交叉熵损失函数适用于分类问题
optimizer = optim.Adam(net.parameters(), lr=0.001)   # 定义优化器，使用Adam优化算法，学习率为0.001
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)   # 定义学习率调度器，当验证损失不再下降时降低学习率

# 训练过程
train_losses = []
test_losses = []
best_f1 = 0
best_model = None
epochs = 100

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device).long()   # 将标签转换为长整型，适用于交叉熵损失函数
        optimizer.zero_grad()   # 清空梯度
        outputs = net(inputs)   # 前向传播
        loss = criterion(outputs, labels)   # 计算损失
        loss.backward()   # 反向传播
        optimizer.step()   # 更新权重
        running_loss += loss.item()   # 累加损失
    
    train_losses.append(running_loss / len(train_loader.dataset))   # 计算平均训练损失
    print(f'[Epoch {epoch+1}/{epochs}] Train Loss: {train_losses[-1]:.4f}')

    # 验证
    net.eval()
    test_loss = 0.0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            outputs = net(inputs)   # 前向传播
            probs = F.softmax(outputs, dim=1)   # 计算预测概率
            loss = criterion(outputs, labels)   # 计算损失
            test_loss += loss.item()   # 累加损失
            all_labels.extend(labels.cpu().numpy())   # 收集真实标签
            all_probs.extend(probs.cpu().numpy())   # 收集预测概率

    test_loss_epoch = test_loss / len(test_loader.dataset)   # 计算平均测试损失
    test_losses.append(test_loss_epoch)
    pred_labels = np.argmax(all_probs, axis=1)   # 获取预测标签
    
    accuracy = accuracy_score(all_labels, pred_labels)   # 计算准确率
    precision = precision_score(all_labels, pred_labels)   # 计算精确率
    recall = recall_score(all_labels, pred_labels)   # 计算召回率
    f1 = f1_score(all_labels, pred_labels)   # 计算F1分数

    print(f'[Epoch {epoch+1}/{epochs}] Test Loss: {test_loss_epoch:.4f}')
    print(f'→ Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    # 动态调整学习率
    scheduler.step(test_loss_epoch)
    current_lr = optimizer.param_groups[0]['lr']
    print(f'→ Current Learning Rate: {current_lr:.6f}')

    if f1 > best_f1:
        torch.save(net.state_dict(), 'best_model.pth')   # 保存当前最佳模型参数到文件
        best_f1 = f1
        best_model = net.state_dict()   # 保存当前最佳模型参数

# 绘制训练和测试损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()
plt.savefig('result/LOSS.png')   # 保存损失曲线图像到文件
plt.close()

# 评估最佳模型
model = CNN().to(device)
model.load_state_dict(torch.load('best_model.pth'))   # 加载最佳模型参数
model.eval()

test_labels = []
test_probs = []

with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)   # 前向传播
        probs = F.softmax(outputs, dim=1)   # 计算预测概率
        test_labels.extend(labels.numpy())   # 收集真实标签
        test_probs.extend(probs.cpu().numpy())   # 收集预测概率

test_labels = np.array(test_labels)
test_probs = np.array(test_probs)
preds = np.argmax(test_probs, axis=1)   # 获取预测标签

accuracy = accuracy_score(test_labels, preds)   # 计算准确率
precision = precision_score(test_labels, preds)   # 计算精确率
recall = recall_score(test_labels, preds)   # 计算召回率
f1 = f1_score(test_labels, preds)   # 计算F1分数
print('\n=== Final Test Performance ===')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1: {f1:.4f}')

# 混淆矩阵
cm = confusion_matrix(test_labels, preds)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0, 1], ['Cat', 'Dog'])
plt.yticks([0, 1], ['Cat', 'Dog'])
plt.savefig('result/Confusion_Matrix.png')   # 保存混淆矩阵图像到文件
plt.close()

# 绘制ROC曲线
fpr0, tpr0, _ = roc_curve(test_labels, test_probs[:, 0])   # 计算类别0的假正率和真正率
fpr1, tpr1, _ = roc_curve(test_labels, test_probs[:, 1])   # 计算类别1的假正率和真正率  
auc0 = auc(fpr0, tpr0)   # 计算类别0的AUC值
auc1 = auc(fpr1, tpr1)   # 计算类别1的AUC值

plt.figure(figsize=(8, 6))
plt.plot(fpr0, tpr0, label=f'Class 0 (Cat) ROC (AUC = {auc0:.2f})', color='blue')   # 绘制类别0的ROC曲线
plt.plot(fpr1, tpr1, label=f'Class 1 (Dog) ROC (AUC = {auc1:.2f})', color='orange')   # 绘制类别1的ROC曲线
plt.plot([0, 1], [0, 1], 'k--')   # 绘制对角线，表示随机猜测的性能
plt.xlabel('False Positive Rate')   # 设置x轴标签
plt.ylabel('True Positive Rate')   # 设置y轴标签
plt.title('ROC Curves for Each Class')   # 设置图表标题
plt.legend(loc='lower right')   # 显示图例
plt.savefig('result/ROC_Curves.png')   # 保存ROC曲线图像到文件
plt.close()