import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path
import re
import json
import urllib.request

# 配置参数
DATA_DIR = "./data/imagenet_compatible/images"  # 图像目录
LABEL_FILE = "./data/imagenet_compatible/labels.txt"  # 标签文件
MAX_IMAGES = 500  # 最多处理图片数量

def get_image_number(path):
    # 提取文件名中的数字，例如 "123.png" -> 123
    return int(re.findall(r'\d+', path.stem)[0])

def load_imagenet_labels():
    # 加载 ImageNet 类别索引映射
    idx2label_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    json_path = "imagenet_class_index.json"
    if not os.path.exists(json_path):
        urllib.request.urlretrieve(idx2label_url, json_path)
    with open(json_path) as f:
        class_idx = json.load(f)
    return {int(k): v[1] for k, v in class_idx.items()}

def main():
    # 1. 加载预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet18(pretrained=True).eval().to(device)
    print(f"Loaded ResNet18 on {device}")

    # 2. 定义预处理流程
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. 加载标签（注意：labels.txt 第1行对应 1.png）
    with open(LABEL_FILE) as f:
        true_labels = [int(line.strip()) for line in f.readlines()]

    # 4. 加载图像路径并按图片编号排序
    image_paths = sorted([
        p for ext in ["*.jpg", "*.jpeg", "*.png"]
        for p in Path(DATA_DIR).rglob(ext)
    ], key=get_image_number)[:MAX_IMAGES]

    if not image_paths:
        raise ValueError(f"No images found in {DATA_DIR}")

    # 5. 加载类别名称（可选）
    idx2label = load_imagenet_labels()

    # 6. 推理与评估
    correct = 0
    total = 0
    results = []

    for img_path in image_paths:
        try:
            img_num = get_image_number(img_path)
            true_label = true_labels[img_num - 1]  # 修复错位！
            true_label = true_label - 1  # 修复错位！

            # 加载并预处理图像
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                output = model(input_tensor)
            pred = torch.argmax(output).item()

            results.append({
                "path": str(img_path),
                "pred": pred,
                "true": true_label,
                "correct": pred == true_label
            })

            if pred == true_label:
                correct += 1
            total += 1

        except Exception as e:
            print(f"处理 {img_path} 失败: {str(e)}")

    # 7. 输出结果
    print("\n分类结果：")
    print(f"总处理图片数: {total}")
    print(f"正确分类数: {correct}")
    print(f"准确率: {correct / total * 100:.2f}%")

    # 输出前5个样本
    print("\n样本示例：")
    for r in results[:5]:
        pred_name = idx2label.get(r['pred'], "Unknown")
        true_name = idx2label.get(r['true'], "Unknown")
        print(f"图像: {Path(r['path']).name}")
        print(f"预测: {r['pred']} ({pred_name}) | 实际: {r['true']} ({true_name}) | 结果: {'✅ 正确' if r['correct'] else '❌ 错误'}\n")

if __name__ == "__main__":
    main()
