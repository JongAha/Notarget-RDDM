import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from src.residual_denoising_diffusion_pytorch import UnetRes, set_seed, Trainer
from src.adversarial_rddm import AdversarialRDDM

# 初始化设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
set_seed(10)

# 配置参数
image_size = 256  # ImageNet图像尺寸
sampling_timesteps = 100  # 采样时间步
train_batch_size = 2
num_samples = 1

# 创建保存目录
os.makedirs('results', exist_ok=True)

# 模型配置
model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_unet=2,
    condition=False,
    objective='pred_res_noise',
    test_res_or_noise="res_noise"
)

# 初始化对抗RDDM模型
diffusion = AdversarialRDDM(
    model,
    image_size=image_size,
    timesteps=1000,
    sampling_timesteps=sampling_timesteps,
    objective='pred_res_noise',
    loss_type='l2',
    condition=False,
    target_model_name='resnet18',
    adv_lambda=0.1  # 对抗损失权重
).cuda()

# 训练器配置
trainer = Trainer(
    diffusion_model=diffusion,
    folder='./data/imagenet-compatible/images',
    train_batch_size=train_batch_size,
    train_lr=2e-4,
    train_num_steps=1000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=False,
    num_samples=num_samples,
    save_and_sample_every=100,
    results_folder='results',
    convert_image_to='RGB',
    condition=False,
    generation=True,
    num_unet=2
)

if __name__ == "__main__":
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 训练后保存模型
    trainer.save(trainer.step)
    print(f"Training completed. Model saved at step {trainer.step}")