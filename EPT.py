import torch
from torch import nn
import torch.nn.functional as F

"""
多年来，主题模型一直在迅速发展，从传统的神经模型到最新的神经模型。然而，现有的主题模型通常在有效性、效率和稳定性方面存在困难，极大地阻碍了其实际应用。
在本文中，我们提出了FASTopic，一种快速、自适应、稳定且可转移的主题模型。FASTopic 遵循一种新的范式：双重语义关系重建 （DSR）。
DSR摒弃了以往传统的基于神经VAE或基于聚类的方法，而是通过对文档、主题和词嵌入之间的语义关系进行建模，通过重建来发现潜在主题。
这带来了一个整洁高效的主题建模框架。我们进一步提出了一种新颖的嵌入运输计划（ETP）方法。ETP不是早期的直接方法，而是明确地将语义关系规范化为最优运输计划。这解决了关系偏差问题，从而导致了有效的主题建模。
对基准数据集的广泛实验表明，与最先进的基线相比，我们的 FASTopic 在各种场景中表现出卓越的有效性、效率、适应性、稳定性和可转移性。
"""

def pairwise_euclidean_distance(x, y):
    cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
    return cost

class ETP(nn.Module):
    def __init__(self, sinkhorn_alpha, init_a_dist=None, init_b_dist=None, OT_max_iter=5000, stopThr=.5e-2):
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
        M = pairwise_euclidean_distance(x, y)
        device = M.device

        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)

        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)

        u = (torch.ones_like(a) / a.size()[0]).to(device) # Kx1

        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = torch.div(b, torch.matmul(K.t(), u) + self.epsilon)
            u = torch.div(a, torch.matmul(K, v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = torch.mul(v, torch.matmul(K.t(), u))
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

        transp = u * (K * v.T)

        loss_ETP = torch.sum(transp * M)

        return loss_ETP, transp


if __name__ == '__main__':
    sinkhorn_alpha = 0.1
    block = ETP(sinkhorn_alpha=sinkhorn_alpha)

    x = torch.rand(10, 5)  # 10 samples with 5 features each
    y = torch.rand(15, 5)  # 15 samples with 5 features each

    loss, transp = block(x, y)

    print("Loss:", loss.item())
    print("Transport matrix shape:", transp.shape)