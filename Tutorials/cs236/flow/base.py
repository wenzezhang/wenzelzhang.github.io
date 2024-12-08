"""
Basic flow model
code ref:https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/core.py
"""
import torch
import torch.nn as nn
import numpy as np

class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution
    """
    def __init__(self, q0, flows, p=None):
       """
       Args:
           q0: 基础分布，简单的易于采样的分布，比如高斯分布
           flows: 可逆变换
           p: 目标分布
       """
       super().__init__()
       self.q0 = q0
       self.flows = nn.ModuleList(flows)
       self.p = p

    def forward(self, z):
        """
        将隐藏变量Z，应用可逆变换变换为x
        Args:
            z: latent variable

        Returns:
            目标分布变量
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        """
        将隐藏变量Z，应用可逆变换变换为x,并且计算对应的雅可比矩阵的行列式
        Args:
            z: 隐藏变量

        Returns:
            目标分布，以及对应的行列式
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det

    def inverse(self, x):
        """
        将x转换为hidden z
        Args:
            x: 从目标分布采样得到的样本x

        Returns:
            z: 对应的隐藏分布z
        """
        for i in range(len(self.flows) -1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """

        Args:
            x:

        Returns:

        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows)-1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x):
        """
        估计前向的KL散度
        Args:
            x: 从目标分布上采样的批样本

        Returns:
            估计的KL散步值
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) -1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return -torch.mean(log_q)

    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q


