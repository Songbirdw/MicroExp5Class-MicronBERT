import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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



# 配置参数
CHECKPOINT_PATH = "best_model.pth"      # 训练好的模型权重路径
IMAGE_FOLDER = "facedata"               # 待识别图片文件夹
CLASS_NAMES = ['disgust', 'happiness', 'repression', 'surprise', 'others']
IMG_SIZE = 224                          # 必须与训练时保持一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理（必须与训练时完全一致）
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])


        

class MicroExpressionRecognizer:
    def __init__(self):
        # 初始化模型结构
        self.model = MicroExpressionClassifier().to(DEVICE)
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        self.model.eval()
        
    def preprocess_image(self, image_path):
        """处理单张图片的完整流程"""
        # 使用PIL保持处理一致性
        pil_img = Image.open(image_path).convert('RGB')
        return transform(pil_img).unsqueeze(0)  # 添加batch维度
        
    def predict(self, image_tensor):
        """执行预测并返回类别名称和置信度"""
        with torch.no_grad():
            outputs = self.model(image_tensor.to(DEVICE))
            probabilities = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)
            return CLASS_NAMES[pred.item()], conf.item()
            
    def visualize_result(self, image_path, pred_class, confidence, save_path):

        # 用OpenCV读取原始图片
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 创建可视化画布（调整画布布局）
        plt.figure(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.15)  # 底部留出文本空间
        
        # 显示原始图片
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("原始图像", fontproperties='SimHei', fontsize=12)
        
        # 显示处理后的输入
        plt.subplot(1, 2, 2)
        processed_img = self._get_processed_for_display(image_path)
        plt.imshow(processed_img)
        plt.axis('off')
        plt.title("预处理输入", fontproperties='SimHei', fontsize=12)
        
        # 添加底部说明文本（支持中文）
        text_str = f"File: {os.path.basename(image_path)}\nResult: {pred_class} (Confidence: {confidence:.2%})"
        plt.figtext(0.5, 0.05, text_str, 
                    ha='center', va='bottom', 
                    fontsize=14, 
                    fontproperties='SimHei',
                    bbox=dict(facecolor='lightyellow', alpha=0.5))
        
        # 保存结果
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()


    def _get_processed_for_display(self, image_path):
        """获取用于显示的预处理图片"""
        pil_img = Image.open(image_path).convert('RGB')
        display_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE)
        ])
        return display_transform(pil_img)

def main():
    # 初始化识别器
    recognizer = MicroExpressionRecognizer()
    
    # 创建结果保存目录
    os.makedirs("results", exist_ok=True)
    
    # 遍历测试图片
    for img_name in os.listdir(IMAGE_FOLDER):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        
        try:
            # 预处理
            input_tensor = recognizer.preprocess_image(img_path)
            
            # 预测
            pred_class, confidence = recognizer.predict(input_tensor)
            
            # 打印结果
            print(f"Image: {img_name}")
            print(f"  Predicted class: {pred_class}")
            print(f"  Confidence: {confidence:.2%}\n")
            
            # 可视化保存
            save_path = os.path.join("results", f"result_{img_name}")
            recognizer.visualize_result(img_path, pred_class, confidence, save_path)
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")

if __name__ == "__main__":
    main()