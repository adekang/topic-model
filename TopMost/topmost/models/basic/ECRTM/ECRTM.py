import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .ECR import ECR


class ECRTM(nn.Module):
    """
        ECRTM 的神经网络模型，这是一个用于主题建模的神经网络，结合了嵌入聚类正则化（Embedding Clustering Regularization, ECR）。
        该模型的设计基于一篇ICML 2023的论文，标题为《Effective Neural Topic Modeling with Embedding Clustering Regularization》。
    """

    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0., pretrained_WE=None, embed_size=200,
                 beta_temp=0.2, weight_loss_ECR=100.0, sinkhorn_alpha=20.0, sinkhorn_max_iter=1000):

        """
        :param vocab_size: 词汇表的大小。
        :param num_topics: 主题数量。
        :param en_units: 编码器的隐藏单元数量。
        :param dropout: dropout 概率。
        :param pretrained_WE: 预训练的词嵌入矩阵。
        :param embed_size: 词嵌入的维度。
        :param beta_temp: 用于控制主题-词分布的平滑程度的温度参数。
        :param weight_loss_ECR: ECR损失的权重。
        :param sinkhorn_alpha:  Sinkhorn算法中的调节参数，控制距离的缩放。
        :param sinkhorn_max_iter:   Sinkhorn算法的最大迭代次数，控制Sinkhorn算法的迭代过程。
        """

        super().__init__()

        # 初始化模型的参数
        self.num_topics = num_topics  # 主题数量
        self.beta_temp = beta_temp  # 温度参数，用于控制主题-词分布的平滑程度。

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)  # 初始化为一个值为1的数组，它用于计算 mu2 和 var2，即先验的均值和方差。
        self.mu2 = nn.Parameter(
            torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))  # mu2 是先验均值，它是一个可训练的参数。
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (
                1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))  # var2 是先验方差，它是一个可训练的参数。

        # mu2 和 var2 的梯度不需要计算，因此设置为不需要梯度。
        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        #  编码器层的定义
        self.fc11 = nn.Linear(vocab_size, en_units)  # 编码器的第一个全连接层。输入维度为词汇表的大小，输出维度为隐藏单元数量。
        self.fc12 = nn.Linear(en_units, en_units)  # 编码器的第二个全连接层。输入维度为隐藏单元数量，输出维度为隐藏单元数量。
        self.fc21 = nn.Linear(en_units, num_topics)  # 编码器的第三个全连接层。输入维度为隐藏单元数量，输出维度为主题数量。
        self.fc22 = nn.Linear(en_units, num_topics)  # 编码器的第四个全连接层。输入维度为隐藏单元数量，输出维度为主题数量。
        self.fc1_dropout = nn.Dropout(dropout)  # 编码器的第一个 dropout 层。
        self.theta_dropout = nn.Dropout(dropout)  # 编码器的第二个 dropout 层。

        # 编码器的 BN 层，用于对编码器的输出进行 BN 处理。
        self.mean_bn = nn.BatchNorm1d(num_topics)  # 主题均值的 BN 层。
        self.mean_bn.weight.requires_grad = False  # 主题均值的 BN 层的权重不需要梯度。
        self.logvar_bn = nn.BatchNorm1d(num_topics)  # 主题方差的 BN 层。
        self.logvar_bn.weight.requires_grad = False  # 主题方差的 BN 层的权重不需要梯度。
        self.decoder_bn = nn.BatchNorm1d(vocab_size,
                                         affine=True)  # 解码器的 BN 层，用于对解码器的输出进行 BN 处理。affine=True 表示 BN 层的参数是可训练的。
        self.decoder_bn.weight.requires_grad = False  # 解码器的 BN 层的权重不需要梯度。

        if pretrained_WE is not None:  # 如果提供了预训练的词嵌入矩阵，则使用预训练的词嵌入矩阵。
            self.word_embeddings = torch.from_numpy(pretrained_WE).float()
        else:
            # 否则，初始化一个随机的词嵌入矩阵。
            self.word_embeddings = nn.init.trunc_normal_(torch.empty(vocab_size, embed_size))

        # 对词嵌入矩阵进行归一化处理。
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))

        # 初始化主题嵌入矩阵。
        self.topic_embeddings = torch.empty((num_topics, self.word_embeddings.shape[1]))
        # 对主题嵌入矩阵进行截断正态分布初始化。目的是为了让主题嵌入矩阵和词嵌入矩阵之间的距离尽可能小。
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        # 对主题嵌入矩阵进行归一化处理。
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))

        # 初始化 ECR 损失函数。 ECR 损失函数是一个自定义的损失函数，用于对主题嵌入矩阵和词嵌入矩阵之间的距禮进行正则化。
        # weight_loss_ECR 是 ECR 损失的权重，sinkhorn_alpha 是 Sinkhorn 算法的调节参数，sinkhorn_max_iter 是 Sinkhorn 算法的最大迭代次数。
        self.ECR = ECR(weight_loss_ECR, sinkhorn_alpha, sinkhorn_max_iter)

    def get_beta(self):
        """
        计算主题-词分布矩阵 beta。
        :return: 主题-词分布矩阵 beta。
        """
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def get_theta(self, input):
        """
        获取主题分布。
        :param input: 输入数据。
        :return: 主题分布。
        """
        theta, loss_KL = self.encode(input)
        if self.training:
            return theta, loss_KL
        else:
            return theta

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧。
        :param mu: 主题均值。
        :param logvar: 主题方差。
        :return:
        """
        if self.training:
            # 训练阶段，使用重参数化技巧。
            std = torch.exp(0.5 * logvar)
            # 生成一个标准正态分布的随机数。
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, input):
        """
        编码器
        :param input:
        :return:
        """
        # 编码器的前向传播过程 softplus -> softplus -> dropout -> BN -> BN
        #   softplus 是一个激活函数，它的公式是 f(x) = ln(1 + e^x)。
        e1 = F.softplus(self.fc11(input))
        e1 = F.softplus(self.fc12(e1))
        # dropout 层，用于防止过拟合。
        e1 = self.fc1_dropout(e1)
        # 主题均值和方差。
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        # 重参数化技巧。
        z = self.reparameterize(mu, logvar)
        # softmax 层，用于将主题均值转换为主题分布。
        #  dim=1 表示在第一个维度上进行 softmax 操作。
        theta = F.softmax(z, dim=1)
        loss_KL = self.compute_loss_KL(mu, logvar)

        return theta, loss_KL

    def compute_loss_KL(self, mu, logvar):
        """
        计算 KL 散度。
        :param mu: 主题均值。
        :param logvar: 主题方差。
        :return:
        """
        var = logvar.exp()  # 将logvar中的每个元素取指数，得到对应的方差。
        var_division = var / self.var2  # 将计算得到的方差除以 self.var2 先验方差。
        diff = mu - self.mu2  # 计算 mu 和 self.mu2 先验均值之间的差。
        diff_term = diff * diff / self.var2  # 将差的平方除以 self.var2 先验方差。
        logvar_division = self.var2.log() - logvar  # 将 self.var2 先验方差的对数减去 logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)  # 计算 KL 散度。
        KLD = KLD.mean()  # 计算 KL 散度的均值。
        return KLD  # 返回 KL 散度。

    def get_loss_ECR(self):
        """
        计算 ECR 损失。
        :return: ECR 损失。
        """
        cost = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        loss_ECR = self.ECR(cost)
        return loss_ECR

    def pairwise_euclidean_distance(self, x, y):
        """
        计算两个矩阵之间的欧氏距离。
        :param x: 第一个矩阵。
        :param y: 第二个矩阵。
        :return: 两个矩阵之间的欧氏距离。
        """
        cost = torch.sum(x ** 2, axis=1, keepdim=True) + torch.sum(y ** 2, dim=1) - 2 * torch.matmul(x, y.t())
        return cost

    def forward(self, input):
        theta, loss_KL = self.encode(input)
        beta = self.get_beta()

        # 计算重构误差
        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)

        # 计算重构损失
        recon_loss = -(input * recon.log()).sum(axis=1).mean()

        # 计算总损失 = 重构损失 + KL 散度
        loss_TM = recon_loss + loss_KL
        # 计算 ECR 损失
        loss_ECR = self.get_loss_ECR()

        # 计算总损失 = 重构损失 + KL 散度 + ECR 损失
        loss = loss_TM + loss_ECR

        rst_dict = {
            'loss': loss,
            'loss_TM': loss_TM,
            'loss_ECR': loss_ECR
        }

        return rst_dict
