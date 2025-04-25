import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.residual_denoising_diffusion_pytorch import UnetRes, set_seed
from src.adversarial_rddm import AdversarialRDDM
import json

# ResNet预处理参数
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]

def save_image_grid(images, path, nrow=4):
    """保存图像网格（范围自动处理）"""
    torchvision.utils.save_image(
        images,
        path,
        nrow=nrow,
        normalize=True,
        scale_each=True
    )

class TestDataset(Dataset):
    def __init__(self, folder, image_size=224, max_images=200):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # 获取以数字命名的文件，例如 1.png, 2.png
        all_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            all_images.extend(Path(folder).rglob(ext))

        # 提取数字并排序
        def get_image_number(p):
            return int(p.stem.split('.')[0])

        all_images = sorted(all_images, key=get_image_number)[:max_images]

        if len(all_images) == 0:
            raise ValueError(f"No images found in {folder}")

        self.image_paths = all_images

        print(f"Found {len(self.image_paths)} images in {folder}")
        print("File extensions found:", set(p.suffix.lower() for p in self.image_paths))

        # 验证标签文件
        label_path = Path(folder).parent / 'labels.txt'
        if not label_path.exists():
            raise FileNotFoundError(f"Label file {label_path} not found")

        with open(label_path, 'r') as f:
            self.labels = [int(line.strip()) - 1 for line in f.readlines()[:max_images]]  # ⚠️ 减1对齐 ImageNet 索引

        print(f"Using {len(self.labels)} labels")

        # 关键修复：直接生成ResNet兼容尺寸 (224x224)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img, self.labels[idx], str(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return torch.zeros(3, 224, 224), self.labels[idx], str(img_path)

def evaluate_attack(target_model, original_images, adv_images, true_labels, device):
    target_model.eval()

    def preprocess(x):
        x = (x + 1) * 0.5  # [-1,1] → [0,1]
        return transforms.Normalize(RESNET_MEAN, RESNET_STD)(x)

    with torch.no_grad():
        orig_processed = preprocess(original_images.to(device))
        adv_processed = preprocess(adv_images.to(device))

        orig_logits = target_model(orig_processed)
        adv_logits = target_model(adv_processed)

        orig_preds = torch.argmax(orig_logits, dim=1)
        adv_preds = torch.argmax(adv_logits, dim=1)

        # 打印每张图像预测信息
        for i in range(len(true_labels)):
            true_label = true_labels[i].item()
            pred_label = orig_preds[i].item()
            print(f"[原始图像预测] 图像 {i}: 真实标签 = {true_label}, 预测标签 = {pred_label}, 结果 = {'✅ 正确' if true_label == pred_label else '❌ 错误'}")

        orig_acc = (orig_preds == true_labels.to(device)).float().mean()
        adv_acc = (adv_preds == true_labels.to(device)).float().mean()
        attack_success = (orig_preds != adv_preds).float().mean()

        return {
            'original_accuracy': orig_acc.item() * 100,
            'adversarial_accuracy': adv_acc.item() * 100,
            'attack_success_rate': attack_success.item() * 100,
            'original_preds': orig_preds.cpu().numpy(),
            'adversarial_preds': adv_preds.cpu().numpy()
        }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    set_seed(42)

    config = {
        'data_path': './data/imagenet_compatible/images',
        'checkpoint_path': './results/model-4.pt',
        'results_dir': './results/attack_results',
        'max_images': 20,
        'batch_size': 1,
        'image_size': 224,
        'sampling_timesteps': 100
    }

    os.makedirs(config['results_dir'], exist_ok=True)

    print("Initializing models...")
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=2,
        condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise"
    ).to(device)

    diffusion = AdversarialRDDM(
        model,
        image_size=config['image_size'],
        timesteps=1000,
        sampling_timesteps=config['sampling_timesteps'],
        objective='pred_res_noise',
        loss_type='l2',
        condition=False,
        target_model_name='resnet18',
        adv_lambda=0.1
    ).to(device)

    print("Loading checkpoint...")
    checkpoint = torch.load(config['checkpoint_path'], map_location=device)
    diffusion.load_state_dict(checkpoint['model'])

    test_dataset = TestDataset(config['data_path'], max_images=config['max_images'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    process_images_dir = os.path.join(config['results_dir'], 'process_images')
    adversarial_dir = os.path.join(config['results_dir'], 'adversarial_samples')
    os.makedirs(process_images_dir, exist_ok=True)
    os.makedirs(adversarial_dir, exist_ok=True)

    all_metrics = []
    total_processed = 0

    resnet18 = torchvision.models.resnet18(pretrained=True).eval().to(device)

    for batch_idx, (images, labels, paths) in enumerate(tqdm(test_loader, desc="Generating adversarial samples")):
        try:
            images = images.to(device)
            labels = labels.to(device)

            adv_images = diffusion.ddim_sample_with_attack(images, labels, images.shape)

            save_path = os.path.join(process_images_dir, f'sample_{total_processed:03d}_process.png')
            save_image_grid(torch.cat([images.cpu(), adv_images.cpu()], dim=0), save_path, nrow=2)

            metrics = evaluate_attack(resnet18, images, adv_images, labels, device)
            all_metrics.append(metrics)

            total_processed += 1

        except Exception as e:
            print(f"\nError processing image {total_processed}: {str(e)}")
            continue

    if all_metrics:
        avg_metrics = {
            'original_accuracy': np.mean([m['original_accuracy'] for m in all_metrics]),
            'adversarial_accuracy': np.mean([m['adversarial_accuracy'] for m in all_metrics]),
            'attack_success_rate': np.mean([m['attack_success_rate'] for m in all_metrics])
        }

        with open(os.path.join(config['results_dir'], 'results.txt'), 'w') as f:
            f.write("Final Results:\n")
            f.write(f"Original Accuracy: {avg_metrics['original_accuracy']:.2f}%\n")
            f.write(f"Adversarial Accuracy: {avg_metrics['adversarial_accuracy']:.2f}%\n")
            f.write(f"Attack Success Rate: {avg_metrics['attack_success_rate']:.2f}%\n")

        print("\nFinal Results:")
        print(f"Original Accuracy: {avg_metrics['original_accuracy']:.2f}%")
        print(f"Adversarial Accuracy: {avg_metrics['adversarial_accuracy']:.2f}%")
        print(f"Attack Success Rate: {avg_metrics['attack_success_rate']:.2f}%")
    else:
        print("No successful evaluations completed")

if __name__ == "__main__":
    main()
