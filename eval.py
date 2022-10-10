from Model import UNet
import torch
from torchvision.utils import save_image
"""参数设置"""
device = 'cuda:0'
T = 1000
beta_1 = 1e-4
beta_T = 0.02
weight_path = 'Checkpoints/ckpt_260.pt'
image_size = 32
"""网络参数设置"""
channel = 128
channel_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.15

device = torch.device(device if torch.cuda.is_available() else 'cpu')  # 设备选择

betas = torch.linspace(beta_1, beta_T, T).to(device)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

model = UNet(T=T, ch=channel, ch_mult=channel_mult, attn=attn,
             num_res_blocks=num_res_blocks, dropout=dropout).to(device)  # 模型
checkpoint = torch.load(weight_path, map_location='cpu')  # 权重加载
model.load_state_dict(checkpoint)
print("model load weights done.")
model.eval()
x_T = torch.randn(1, 3, image_size, image_size).to(device)  # 采样自标准正态分布的x_T
x_t = x_T
with torch.no_grad():
    for t_step in reversed(range(T)):  # 从T开始向零迭代
        t = t_step
        t = torch.tensor(t).to(device)

        z = torch.randn_like(x_t, device=device) if t_step > 0 else 0  # 如果t大于零，则采样自标准正态分布，否则为零
        """按照公式计算x_{t-1}"""
        x_t_minus_one = torch.sqrt(1 / alphas[t]) * (
                    x_t - (1 - alphas[t]) * model(x_t, t.reshape(1, )) / torch.sqrt(1 - alphas_bar[t])) + torch.sqrt(
            betas[t]) * z

        x_t = x_t_minus_one
        print(t_step)
    # x_0 = torch.clip(x_t,-1,1)

    x_0 = x_t
    x_0 = x_0 * 0.5 + 0.5
    save_image(x_0, 'sample.jpg')
