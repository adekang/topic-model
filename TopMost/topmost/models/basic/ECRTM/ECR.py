import torch
from torch import nn


class ECR(nn.Module):
    '''
        Effective Neural Topic Modeling with Embedding Clustering Regularization. ICML 2023

        Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu.
    '''

    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=5000, stopThr=.5e-2):
        super().__init__()

        self.sinkhorn_alpha = sinkhorn_alpha  # Sinkhorn算法中的调节参数，控制距离的缩放。
        self.OT_max_iter = OT_max_iter  # Sinkhorn算法的最大迭代次数，控制Sinkhorn算法的迭代过程。
        self.weight_loss_ECR = weight_loss_ECR  # ECR损失的权重，决定其对总损失的影响。
        self.stopThr = stopThr  # Sinkhorn算法的停止阈值，控制Sinkhorn算法的迭代过程。
        self.epsilon = 1e-16  # Sinkhorn算法中的一个小常数，防止除零错误。

    def forward(self, M):
        # M: KxV  M 是输入矩阵，表示主题嵌入和词嵌入之间的距离矩阵，大小为 KxV，其中 K 是主题数量，V 是词汇表的大小。
        # a: Kx1 a 是主题分布的先验分布，大小为 Kx1。
        # b: Vx1 b 是词分布的先验分布，大小为 Vx1。
        device = M.device

        # Sinkhorn's algorithm
        # 分别表示主题和词汇表的概率分布，初始为均匀分布，并扩展为列向量。
        a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device) # Kx1
        b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device) # Vx1

        u = (torch.ones_like(a) / a.size()[0]).to(device)  # Kx1
        K = torch.exp(-M * self.sinkhorn_alpha) # K 是经过缩放的距离矩阵的指数形式，用于在迭代过程中计算传输计划。
        err = 1 # 误差
        cpt = 0 # 迭代次数
        while err > self.stopThr and cpt < self.OT_max_iter: # 迭代停止条件
            # v,u 是Sinkhorn算法中的辅助变量，用于计算传输计划矩阵 transp
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)

        loss_ECR = torch.sum(transp * M)
        loss_ECR *= self.weight_loss_ECR
        # loss_ECR 是ECR损失的最终计算结果，并在前向传播中返回，作为正则化损失的一部分，用于训练神经网络模型。
        return loss_ECR
