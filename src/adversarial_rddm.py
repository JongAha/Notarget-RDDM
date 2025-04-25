import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from dataclasses import dataclass
from src.residual_denoising_diffusion_pytorch import ResidualDiffusion


@dataclass
class ModelPrediction:
    """用于存储模型预测结果的数据类"""
    pred_noise: torch.Tensor
    pred_res: torch.Tensor
    pred_x_start: torch.Tensor


class AdversarialRDDM(ResidualDiffusion):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            loss_type='l1',
            condition=False,
            sum_scale=1,
            input_condition=False,
            input_condition_mask=False,
            test_res_or_noise="res_noise",
            adv_lambda=0.1,
            target_model_name='resnet18'
    ):
        super().__init__(
            model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective=objective,
            loss_type=loss_type,
            condition=condition,
            sum_scale=sum_scale,
            input_condition=input_condition,
            input_condition_mask=input_condition_mask,
            test_res_or_noise=test_res_or_noise
        )

        self.adv_lambda = adv_lambda
        self.test_mode = test_res_or_noise
        self.objective = objective

        # 初始化目标模型
        if target_model_name == 'resnet18':
            self.target_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.target_model = models.__dict__[target_model_name](weights=True)

        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_model = self.target_model.cuda()

    def compute_adv_loss(self, x, y_true):
        """计算对抗损失"""
        logits = self.target_model(x)
        return -F.cross_entropy(logits, y_true)

    def forward(self, x, y_true=None):
        """修改前向传播以包含对抗损失"""
        try:
            b, c, h, w = x.shape
            t = torch.randint(0, self.num_timesteps, (b,), device=x.device).long()

            # 计算残差
            x_res = x

            # 生成噪声
            noise = torch.randn_like(x_res)

            # 使用正确的参数调用 q_sample
            x_noisy = self.q_sample(x_start=x, t=t, x_res=x_res)

            # 根据模型配置处理输出
            model_output = self.model(x_noisy, t)

            if isinstance(model_output, (list, tuple)):
                if len(model_output) == 2:
                    pred_res, pred_noise = model_output
                else:
                    pred_res = model_output[0]
                    pred_noise = torch.zeros_like(pred_res)
            else:
                pred_res = model_output
                pred_noise = torch.zeros_like(pred_res)

            # 计算基础损失
            losses = []
            if self.objective == 'pred_res_noise':
                losses.append(F.mse_loss(pred_res, x_res))
                losses.append(F.mse_loss(pred_noise, noise))
            else:
                losses.append(F.mse_loss(pred_res, x_res))

            # 如果提供了标签，添加对抗损失
            if y_true is not None:
                pred_x_start = x + pred_res
                logits = self.target_model(pred_x_start)
                adv_loss = -F.cross_entropy(logits, y_true)

                # 将对抗损失添加到每个损失中
                losses = [loss + self.adv_lambda * adv_loss for loss in losses]

            return losses

        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

    def ddim_sample_with_attack(self, x_input, y_true, shape, callback=None):
        device = self.betas.device

        x_input = x_input.to(device)
        y_true = y_true.to(device)

        batch = shape[0]
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1,
                             steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device) if not self.condition else x_input + (self.sum_scale ** 0.5) * torch.randn(shape, device=device)

        if callback is not None:
            callback(0, img)

        for step, (time, time_next) in enumerate(time_pairs):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            with torch.enable_grad():
                img.requires_grad_(True)

                model_output = self.model(img, time_cond)

                if isinstance(model_output, (list, tuple)):
                    pred_res, pred_noise = model_output
                else:
                    pred_res = model_output
                    pred_noise = torch.zeros_like(pred_res)

                x_start = x_input + pred_res

                logits = self.target_model(x_start)
                adv_loss = -F.cross_entropy(logits, y_true)
                grad = torch.autograd.grad(adv_loss, img)[0]

                # 释放不需要的内存
                del logits, adv_loss

            img = img.detach() + self.adv_lambda * grad.sign()

            alpha = (1 - self.betas[time]).sqrt()
            alpha_next = (1 - self.betas[time_next]).sqrt() if time_next >= 0 else torch.tensor(1.)
            sigma = eta * ((1 - alpha_next ** 2) / (1 - alpha ** 2)).sqrt() * (1 - alpha ** 2).sqrt()

            if time_next < 0:
                img = x_start
            else:
                c1 = (1 - alpha_next ** 2 - sigma ** 2).sqrt()
                c2 = (1 - alpha_next ** 2 - sigma ** 2).sqrt() * alpha ** 2 / alpha_next
                c3 = sigma * alpha
                noise = torch.randn_like(img) if sigma > 0 else 0.
                img = c1 * x_start + c2 * img + c3 * noise

            img = torch.clamp(img, -1, 1)

            if callback is not None:
                callback(step + 1, img.detach())

            # 释放不需要的内存
            del grad, x_start, pred_res, pred_noise
            torch.cuda.empty_cache()

        return img