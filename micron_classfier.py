import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from functools import partial
import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from functools import partial


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def image_to_tensor(image):
    """将numpy数组转为模型输入张量"""
    x = torch.tensor(image).float()
    x = x.permute(2, 0, 1)          # HWC -> CHW
    x = x.unsqueeze(0)               # 添加batch维度 [1,3,224,224]
    return x.cuda()

def preprocess(image, img_size=224):
    """OpenCV图像预处理"""
    # 缩放到224x224
    image = cv2.resize(image, (img_size, img_size))
    # 转换颜色空间 [0,255] -> [0,1]
    image = image.astype(np.float32) / 255.0
    # 标准化
    image = (image - imagenet_mean) / imagenet_std
    return image

def load_image(image_path):
    """加载单张测试图片"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR转RGB
    return preprocess(image)

def get_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    args = checkpoint["args"]
    print(args)
    print(checkpoint["epoch"])

    import models.mae as mae_dict

    if "mae" in args.model_name:
        import models.mae as mae_dict

        model = mae_dict.__dict__[args.model_name](
            has_decoder=args.has_decoder,
            aux_cls=args.aux_cls,
            img_size=args.img_size,
            att_loss=args.att_loss,
            diag_att=args.diag_att,
            # DINO params,
            enable_dino=args.enable_dino,
            out_dim=args.out_dim,
            local_crops_number=args.local_crops_number,
            warmup_teacher_temp=args.warmup_teacher_temp,
            teacher_temp=args.teacher_temp,
            warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
            epochs=args.epochs,
        )

    model_state_dict = {}
    for k, v in checkpoint["model"].items():
        model_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(model_state_dict)
    return model, args

def load_micron_bert_model():
    # checkpoint = 'checkpoints/CASME2-is224-p8-b16-ep200.pth'
    checkpoint = 'CASME2-is224-p8-b16-ep200.pth'
    # your_image_path = 'image.jpg'
    model = get_model(checkpoint)[0].cuda()
    # model.eval()
    return model

class MicroExpressionClassifier(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        """
        微表情分类模型
        :param num_classes: 类别数（根据你的emote_dict设为5）
        :param freeze_backbone: 是否冻结预训练骨干网络
        """
        super().__init__()
        
        # 加载预训练特征提取器
        self.backbone = load_micron_bert_model()
        
        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
                
        # 分类头设计
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),       # 标准化
            nn.Dropout(0.3),         # 防止过拟合
            nn.Linear(512, 256),     # 第一个全连接层
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)  # 最终分类层
        )
        
        # 初始化分类头参数
        self._init_weights(self.classifier)
        
    def _init_weights(self, module):
        """参数初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        """
        前向传播
        :param x: 输入张量 [batch_size, 3, 224, 224]
        :return: 分类logits [batch_size, num_classes]
        """
        # 提取特征 [batch_size, 784, 512]
        features = self.backbone.extract_features(x)
        
        # 全局平均池化 [batch_size, 512]
        pooled = features.mean(dim=1)
        
        # 分类预测 [batch_size, num_classes]
        return self.classifier(pooled)


# 训练函数
def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)
    print(f'\nTrain set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({acc:.2f}%)\n')
    return avg_loss, acc

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 数据路径配置
data_root = "CASME2 Preprocessed v2"  # 根据实际路径修改
class_names = ['disgust', 'happiness', 'repression', 'surprise', 'others']

# 数据预处理
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])





# 新增依赖
import math
from sklearn.metrics import accuracy_score, recall_score
from matplotlib.backends.backend_pdf import PdfPages

# ==================== 数据划分优化 ====================
# 全局划分数据集（6:2:2）
full_dataset = ImageFolder(data_root)
all_indices = list(range(len(full_dataset)))

# 首次划分：80%训练验证 + 20%测试
trainval_idx, test_idx = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=full_dataset.targets,
    random_state=42
)





# 二次划分：训练60% + 验证20%
train_idx, val_idx = train_test_split(
    trainval_idx,
    test_size=0.25,  # 0.25 * 0.8 = 0.2
    stratify=[full_dataset.targets[i] for i in trainval_idx],
    random_state=42
)

class SplitDataset(Dataset):
    def __init__(self, main_dir, indices, transform=None):
        self.full_dataset = ImageFolder(main_dir)
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.full_dataset[real_idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 创建数据集
train_dataset = SplitDataset(data_root, train_idx, transform=train_transform)
val_dataset = SplitDataset(data_root, val_idx, transform=test_transform)
test_dataset = SplitDataset(data_root, test_idx, transform=test_transform)

# 更新数据加载器
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, num_workers=4)

# 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MicroExpressionClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

# ==================== 增强版指标记录器 ====================
class EnhancedMetricLogger:
    def __init__(self, class_names):
        self.class_names = class_names
        self.metrics = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def update(self, epoch, t_loss, t_acc, v_loss, v_acc):
        """更新指标并生成实时图表"""
        self.metrics['train_loss'].append(t_loss)
        self.metrics['train_acc'].append(t_acc)
        self.metrics['val_loss'].append(v_loss)
        self.metrics['val_acc'].append(v_acc)
        
        # 实时更新训练曲线
        self._plot_training_curve(epoch)
        
    def _plot_training_curve(self, epoch):
        """生成训练指标曲线图"""
        plt.figure(figsize=(12, 5))
        
        # Loss曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['train_loss'], 'b-', label='Train')
        plt.plot(self.metrics['val_loss'], 'r-', label='Validation')
        plt.title(f'Loss Curve (Epoch {epoch})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['train_acc'], 'b-', label='Train')
        plt.plot(self.metrics['val_acc'], 'r-', label='Validation')
        plt.title(f'Accuracy Curve (Epoch {epoch})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_metrics_epoch{epoch:03d}.png')
        plt.close()

    def save_radar_chart(self, recalls, epoch):
        """生成类别精度雷达图"""
        angles = np.linspace(0, 2 * np.pi, len(recalls), endpoint=False).tolist()
        angles += angles[:1]  # 闭合曲线
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # 绘制填充区域
        values = recalls + recalls[:1]
        ax.fill(angles, values, 'b', alpha=0.2)
        
        # 添加类别标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.class_names, fontsize=10)
        
        # 设置极坐标网格
        ax.set_rlabel_position(30)
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=8)
        plt.ylim(0, 1)
        
        plt.title(f'Class Recall Radar (Epoch {epoch})', pad=20)
        plt.savefig(f'radar_chart_epoch{epoch:03d}.png')
        plt.close()

# ==================== 改进验证函数 ====================
def evaluate(data_loader, epoch=0, mode='val'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算指标
    acc = accuracy_score(all_targets, all_preds)
    cls_report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_targets, all_preds)
    
    # 保存混淆矩阵
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title(f'Confusion Matrix ({mode} Epoch {epoch})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # 处理epoch格式
    epoch_str = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    plt.savefig(f'confusion_matrix_{mode}_epoch{epoch_str}.png')
    plt.close()
    
    # 提取各类别召回率
    recalls = [cls_report[cls]['recall'] for cls in class_names]
    
    return total_loss / len(data_loader), acc * 100, recalls

# ==================== 主训练循环优化 ====================
logger = EnhancedMetricLogger(class_names)
best_acc = 0

for epoch in range(1, 51):
    # 训练阶段
    train_loss, train_acc = train(epoch)
    
    # 验证阶段
    val_loss, val_acc, recalls = evaluate(val_loader, epoch, 'val')
    
    # 更新日志并生成图表
    logger.update(epoch, train_loss, train_acc, val_loss, val_acc)
    logger.save_radar_chart(recalls, epoch)
    
    # 学习率调整
    scheduler.step(val_acc)
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Saved new best model with val_acc {val_acc:.2f}%")

# 最终测试
print("\nFinal Test Results:")
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_acc, _ = evaluate(test_loader, epoch='final', mode='test')
print(f"\nTest Accuracy: {test_acc:.2f}%")