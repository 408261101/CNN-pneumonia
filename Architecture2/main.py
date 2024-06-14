import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class PneumoniaDataset(Dataset):
    def __init__(self, normal_dir, pneumonia_dir, transform=None):
        self.normal_images = [os.path.join(normal_dir, img) for img in os.listdir(normal_dir)]
        self.pneumonia_images = [os.path.join(pneumonia_dir, img) for img in os.listdir(pneumonia_dir)]
        self.all_images = self.normal_images + self.pneumonia_images
        # 正常標籤為 0，肺炎標籤為 1
        self.labels = [0] * len(self.normal_images) + [1] * len(self.pneumonia_images)
        self.transform = transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_path = self.all_images[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

# 指定轉換
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 創建數據集
dataset = PneumoniaDataset(normal_dir='train/NORMAL',
                           pneumonia_dir='train/PNEUMONIA',
                           transform=transform)

# 創建數據加載器
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用相同的轉換創建測試數據集實例
test_dataset = PneumoniaDataset(normal_dir='test/NORMAL',
                                pneumonia_dir='test/PNEUMONIA',
                                transform=transform)
# 創建測試數據加載器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定義深度CNN模型
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()
        # 定義卷積層
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # 定義池化層
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定義全連接層
        # 調整全連接層的输入特征数
        self.fc1 = nn.Linear(in_features=256 * 64 * 64, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2)  # 有2個输出類别

    def forward(self, x):
        # 通過卷積層和激活層
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # 第一次池化后尺寸變為 128x128
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # 第二次池化后尺寸變為 64x64
        x = self.pool(x)
        # 展平特徵圖以便输入到全連接層
        x = x.view(-1, 256 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 創建模型實例
model = DeepCNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 計算總參數量
total_params = sum(p.numel() for p in model.parameters())

# 計算卷積層參數量
conv_params = sum(p.numel() for layer in [model.conv1, model.conv2, model.conv3, model.conv4] for p in layer.parameters())

# 計算全連接層參數量
fc_params = sum(p.numel() for layer in [model.fc1, model.fc2, model.fc3] for p in layer.parameters())

print(f"總參數量: {total_params}")
print(f"卷積層參數量: {conv_params}")
print(f"全連接層參數量: {fc_params}")


#存放損失和準確率，供後續繪圖使用
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # 設置模型到訓練模式
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        # 正常的前向傳播、損失計算、反向傳播和優化步驟
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加損失值
        running_loss += loss.item()

        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    #計算訓練準確率
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    # 計算平均損失並添加到訓練損失列表中
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    # 驗證過程，這裡假設您有 val_loader 用於驗證
    model.eval()  # 設置模型到評估模式
    running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 累加損失值
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    #計算驗證準確率
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)
    # 計算平均損失並添加到驗證損失列表中
    epoch_loss = running_loss / len(test_loader)
    val_losses.append(epoch_loss)

    # 打印準確率和損失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

# 繪製損失曲線圖
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
plt.show()

# 繪製準確率曲線
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_curve.png')
plt.show()