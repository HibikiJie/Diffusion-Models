# Diffusion Models

**扩散模型**是受非平衡热力学的启发。它们定义一个扩散步骤的马尔可夫链，逐渐向数据添加随机噪声，然后学习逆扩散过程，从噪声中构建所需的数据样本。与VAE或流动模型不同，扩散模型是用固定的程序学习的，而且隐变量具有高维度。

训练阶段，是在图片中添加噪声，给网络输入这一张添加噪声的图片，网络需要预测的则是添加的噪声。

使用阶段，由随机生成的噪声，使用网络预测添加了什么噪声，然后逐步去除噪声，直到还原。



# 1、扩散过程

- 扩散的过程，是不断地逐步向图片中添加噪声，直到图像完全变为纯噪声；
- 添加的噪声为高斯噪声，而后一时刻都是由前一时刻的图像增加噪声得到的。

**添加噪声的过程：**

这里定义了两个参数$\alpha_t$、$\beta_t$；（$t$的范围为0~$T$之间的整数，$\beta_t$ 从$\beta_1$变化到$\beta_T$,是逐渐变大的，论文中是从0.0001等量的增加$T$次，直到0.002  ）

而 $\alpha_t$与$\beta_t$的关系为：

$\alpha_t = 1- \beta_t$     								（1)

$T$表示的是，由图片通过逐步添加噪声直至完全变为纯噪声的过程所需要经历的次数，也就是图片需要总计需要添加噪声的步数。而$t$则代表的是$T$中具体的某一步。

则给图像添加噪声的过程的表达式可以写为：

$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}z_t$       	  (2)

$x_t$表示的是第$t$扩散步时，添加噪声后的图片，而$x_{t-1}$是第$t-1$时刻所获得的图片；$z_t$表示的是$t$时刻所添加的噪声，该噪声采样自标准正态分布$N(0,1)$

那么可以依照公式(2)依次从原始图像$x_0$逐步添加噪声，扩散至$x_T$：

$x_1=\sqrt{\alpha_1}x_{0}+\sqrt{1-\alpha_1}z_1$

$x_2=\sqrt{\alpha_2}x_{1}+\sqrt{1-\alpha_2}z_2$

……

$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}z_t$ 

……

$x_T=\sqrt{\alpha_T}x_{T-1}+\sqrt{1-\alpha_T}z_T$ 

由此可以看出$\beta_t$逐渐增加，相应的$\alpha_t$逐渐减小，$1-\alpha_t$则是逐渐增大的，也就是说，添加的噪声是逐步增加的，而原始图像的比例是逐渐减小的,并且噪声添加的程度是逐次扩大的。

但对网络的训练，数据是需要随机采样的，每次采样到$t$时刻的时候，都从$x_0$开始递推则太过于繁琐。

所以需要一次就计算出来：

将式 ：$x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}z_{t-1}$带入（2）式中，可得

$x_t=\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}z_{t-1})+\sqrt{1-\alpha_t}z_t$  

式子展开为：

$x_t=\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}}z_{t-1}+\sqrt{1-\alpha_t}z_t$  

$=\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}x_{t-2}+(\sqrt{\alpha_t (1- \alpha_{t-1})}z_{t-1}+\sqrt{1-\alpha_t}z_t)$  

其中每次加入的噪声——$z_1,z_2,...,z_{t-1},z_t,...z_T$——都是服从正态分布 $N(0,1)$

 所以可以将

$z_{t-1}$和$z_t$之间的系数合并在一起，因为正太分布乘以一个系数，只改变方差，而$N(0,\sigma_1^2)+N(0,\sigma_2^2)~N(0,\sigma_1^2+\sigma_2^2 )$

所以

$x_t=\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}x_{t-2}+(\sqrt{\alpha_t (1- \alpha_{t-1})}z_{t-1}+\sqrt{1-\alpha_t}z_t)$  

$=\sqrt{\alpha_t}\sqrt{\alpha_{t-1}}x_{t-2}+(\sqrt{a_t(1-a_{t-1})+1-a_t})z$  

$=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+(\sqrt{1-a_t a_{t-1}})z$  

再将$x_{t-2}=\sqrt{\alpha_{t-2}}x_{t-3}+\sqrt{1-\alpha_{t-2}}z_{t-2}$带入上式，循环往复，将$x_1$带入，可得

$x_t=\sqrt{\alpha_t\alpha_{t-1}...\alpha_2 \alpha_1}x_{t-1}+(\sqrt{1-\alpha_t\alpha_{t-1}...\alpha_2 \alpha_1})z$  

$=\sqrt{\overline{\alpha_t}}x_{0}+(\sqrt{1-\overline{a_t}})z$    （3）

其中$\overline{\alpha_t}$表示从$\alpha_1$到$\alpha_t$的连乘



# 2、训练过程

因此，扩散模型的训练过程如下：

![image-20221010195212379](../../CStudy/image/image-20221010195212379.png)

1. 从数据集中随机抽选一张图片，
2. 随机从1~T中抽取一个扩散步，
3. 按照式（3）计算得到 $x_t$，
4. 输入网络，得到输出，输出同添加的噪声做损失，更新梯度，
5. 反复训练，直至满意。

详细训练过程的代码过程如下：

```python
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
```



# 3、正向使用过程

使用过程是从$x_T$一步一步取出噪声，推测出$x_0$

也就是说，需要在已知$x_T$的情况下，先反推$x_{t-1}$，然后推$x_{t-2}$……最终推测得到$x_0$

根据贝叶斯公式推导为：
$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha_t}}}M(x_t, t))+\sqrt{\beta_t}z
$$
则整个算法为：

![image-20221010203840101](../../CStudy/image/image-20221010203840101.png)

1. $x_T$随机采样自标准正态分布；
2. 从T到1开始循环，
3. 按照上述公式计算$x_{t-1}$，依次往复，其中$M(x_t, t)$为网络模型，输入的是$x_t$步的结果和第t步，因为模型要对每一步的位置进行编码，$z$取样至标准正态分布，在t为最后一步的时候，z取零

具体代码如下：

```python
for t_step in reversed(range(T)):  # 从T开始向零迭代
    t = t_step
    t = torch.tensor(t).to(device)

    z = torch.randn_like(x_t,device=device) if t_step > 0 else 0  # 如果t大于零，则采样自标准正态分布，否则为零
    """按照公式计算x_{t-1}"""
    x_t_minus_one = torch.sqrt(1/alphas[t])*(x_t-(1-alphas[t])*model(x_t, t.reshape(1,))/torch.sqrt(1-alphas_bar[t]))+torch.sqrt(betas[t])*z
    
    x_t = x_t_minus_one
```



# 4、结果

因为设备有限，训练个网红人脸数据，网络生成的结果如下：

# ![image-20221010204754313](../../CStudy/image/image-20221010204754313.png)



# 5、网络模型

模型使用UNet，并具有第t扩散步的位置编码信息。



# 6、其他

github：