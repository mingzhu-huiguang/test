import os
import random
import shutil
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from tqdm import tqdm
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, real_dir="real", fake_dir="fake", train_size=500):
    real_imgs = [os.path.join(data_dir, real_dir, img) for img in os.listdir(os.path.join(data_dir, real_dir))]
    fake_imgs = [os.path.join(data_dir, fake_dir, img) for img in os.listdir(os.path.join(data_dir, fake_dir))]

    # 随机选取500张真实和500张虚假作为训练集
    train_real = random.sample(real_imgs, train_size)
    train_fake = random.sample(fake_imgs, train_size)
    test_real = [img for img in real_imgs if img not in train_real]
    test_fake = [img for img in fake_imgs if img not in train_fake]

    # 创建训练集和测试集目录
    os.makedirs(os.path.join(data_dir, "train", "real"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "train", "fake"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test", "real"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test", "fake"), exist_ok=True)

    # 移动文件
    for img in train_real:
        shutil.copy(img, os.path.join(data_dir, "train", "real"))
    for img in train_fake:
        shutil.copy(img, os.path.join(data_dir, "train", "fake"))
    for img in test_real:
        shutil.copy(img, os.path.join(data_dir, "test", "real"))
    for img in test_fake:
        shutil.copy(img, os.path.join(data_dir, "test", "fake"))

# 调用函数划分数据集
split_dataset("")


# 数据增强和归一化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_data = datasets.ImageFolder("train", transform=train_transform)
test_data = datasets.ImageFolder("test", transform=test_transform)

# 数据加载器
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


class FakeFaceDetector(nn.Module):
    def __init__(self):
        super(FakeFaceDetector, self).__init__()
        self.resnet = models.resnet18(pretrained=True)  # 加载预训练模型
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)  # 修改最后一层为二分类

    def forward(self, x):
        return self.resnet(x)

model = FakeFaceDetector()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100*correct/total:.2f}%")

train(model, train_loader, criterion, optimizer, epochs=10)

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100*correct/total:.2f}%")

test(model, test_loader)