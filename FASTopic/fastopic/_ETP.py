import torch
from torch import nn
import torch.nn.functional as F
from ._model_utils import pairwise_euclidean_distance


class ETP(nn.Module):
    def __init__(self, sinkhorn_alpha, init_a_dist=None, init_b_dist=None, OT_max_iter=5000, stopThr=.5e-2):
        """
        Args:
            sinkhorn_alpha: Sinkhorn算法中的缩放参数，控制了传输矩阵的平滑性
            init_a_dist: 分别是两个分布的初始值（可选），如果未提供，则会在后续步骤中使用均匀分布。
            init_b_dist:
            OT_max_iter: 最大迭代次数，用于控制Sinkhorn算法的停止条件
            stopThr: 用于数值稳定性，防止除零错误。
        """
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.stopThr = stopThr
        self.epsilon = 1e-16
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist

        if init_a_dist is not None:
            self.a_dist = init_a_dist

        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, x, y):
        # Sinkhorn's algorithm
        M = pairwise_euclidean_distance(x, y) # 欧几里得距离
        device = M.device

        if self.init_a_dist is None:
            # 默认分布为均匀分布（每个点的权重相等）。
            a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        else:
            # 通过 F.softmax 进行归一化，以确保分布的总和为1。
            a = F.softmax(self.a_dist, dim=0).to(device)

        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)

        # 初始化为一个均匀分布的向量
        u = (torch.ones_like(a) / a.size()[0]).to(device) # Kx1

        # 传输矩阵 K
        K = torch.exp(-M * self.sinkhorn_alpha) # 指数
        err = 1 # 误差，用于跟踪算法收敛情况。
        cpt = 0 # 迭代次数计数器。

        # 这是Sinkhorn算法的核心迭代步骤，用于调整向量 u 和 v 以逼近最优解：
        while err > self.stopThr and cpt < self.OT_max_iter:
            # 更新 v，使得其符合目标分布 b
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon) # torch.div 数组的’点除’运算
            # 更新 u，使得其符合目标分布 a
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                # 每50次迭代，计算误差 err
                # 即检查当前传输矩阵与目标分布 b 的差异，如果误差小于阈值，算法停止。
                bb = torch.mul(v, torch.matmul(K.t(), u)) # 矩阵点乘运算
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        # 最终的最优传输矩阵
        transp = u * (K * v.T)
        # 最优传输损失
        loss_ETP = torch.sum(transp * M)

        return loss_ETP, transp
