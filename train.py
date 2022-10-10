import time
import torch
import tqdm
from dataset import FaceDataset
from torch.utils.data import DataLoader
# from unet import UNet
from Model import UNet
from matplotlib import pyplot as plt
"""参数设置"""
epochs = 1000
batch_size = 80
T = 1000
learn_rate = 0.0001
beta_1 = 1e-4
beta_T = 0.02

"""网络参数设置"""
image_size = 32
device = 'cuda:0'
save_weight_dir = 'Checkpoints'
channel = 128
channel_mult = [1, 2, 3, 4]
attn = [2]
num_res_blocks = 2
dropout = 0.15
num_workers = 1  # 数据加载核心数

betas = torch.linspace(beta_1, beta_T, T)  # betas
alphas = 1 - betas  # alphas
alphas_bar = torch.cumprod(alphas, dim=0)  # alpha一把
sqrt_alphas_bar = torch.sqrt(alphas_bar).to(device)  # 根号下alpha一把
sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar).to(device)

"""training before"""
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
dataset = FaceDataset(path='data',size=image_size)
data_loader = DataLoader(dataset, batch_size, True, num_workers=num_workers, drop_last=True)
net_model = UNet(T=T, ch=channel, ch_mult=channel_mult, attn=attn,
                 num_res_blocks=num_res_blocks, dropout=dropout).to(device)
loss_function = torch.nn.MSELoss()

optimizer = torch.optim.AdamW(net_model.parameters(), lr=learn_rate, weight_decay=1e-4)
if __name__ == '__main__':

    """training"""
    for epoch in range(epochs):
        loss_sum = 0
        with tqdm.tqdm(data_loader) as tqdm_data_loader:
            "---------------------"
            for i, (x_0) in enumerate(tqdm_data_loader):  # 由数据加载器加载数据，
                x_0 = x_0.to(device)  # 将数据加载至相应的运行设备(device)
                t = torch.randint(1, T, size=(x_0.shape[0],), device=device)  # 对每一张图片随机在1~T的扩散步中进行采样
                sqrt_alpha_t_bar = torch.gather(sqrt_alphas_bar, dim=0, index=t).reshape(-1, 1, 1, 1)  # 取得不同t下的 根号下alpha_t的连乘
                """取得不同t下的 根号下的一减alpha_t的连乘"""
                sqrt_one_minus_alpha_t_bar = torch.gather(sqrt_one_minus_alphas_bar, dim=0, index=t).reshape(-1, 1, 1, 1)
                noise = torch.randn_like(x_0).to(device)  # 从标准正态分布中采样得到z
                x_t = sqrt_alpha_t_bar * x_0 + sqrt_one_minus_alpha_t_bar * noise  # 计算x_t
                out = net_model(x_t, t)  # 将x_t输入模型，得到输出
                loss = loss_function(out, noise) # 将模型的输出，同添加的噪声做损失
                optimizer.zero_grad()  # 优化器的梯度清零
                loss.backward()  # 由损失反向求导
                optimizer.step()  # 优化器更新参数
                "---------------------"
                tqdm_data_loader.set_description(f"Epoch:{epoch}")
                loss_sum+=loss.item()
                tqdm_data_loader.set_postfix(ordered_dict={
                    "batch": f"{i}/{len(tqdm_data_loader)}",
                    "loss": loss_sum/(i+1)*10000,
                })
                time.sleep(0.1)
            if epoch %10==0:
                torch.save(net_model.state_dict(), f'{save_weight_dir}/ckpt_{epoch}.pt')  # 保存模型参数
