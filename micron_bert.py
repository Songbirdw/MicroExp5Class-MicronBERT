import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
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

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = load_micron_bert_model()
    
    # 处理测试图片
    image_path = "data/cat.jpg"
    image = load_image(image_path)                   # 预处理 [224,224,3]
    image_tensor = image_to_tensor(image)            # 转为张量 [1,3,224,224]
    
    # 提取特征
    with torch.no_grad():
        features = model.extract_features(image_tensor)  # 输出形状 [1, 784, 512]
    
    # 全局平均池化获取图像特征向量
    feature_vector = features.mean(dim=1)            # [1, 512]
    print("特征向量形状:", feature_vector.shape)