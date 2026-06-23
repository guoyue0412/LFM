"""MoE 气动代理模型 — Mixture of Experts + 不确定度输出

架构:
    Input(47) → SharedEncoder(47→64) → GatingNetwork(64→K) + K×Expert(64→8)
    每个 Expert 输出: [μ_T, μ_H, μ_My, μ_Q, log_σ_T, log_σ_H, log_σ_My, log_σ_Q]
    最终输出: 混合均值 μ 和混合方差 σ²
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertNetwork(nn.Module):
    """单个 Expert: 预测均值和 log 方差。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim // 2, output_dim)
        self.log_sigma_head = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        log_sigma = torch.clamp(log_sigma, min=-6.0, max=2.0)
        return mu, log_sigma


class GatingNetwork(nn.Module):
    """门控网络: 决定每个 Expert 的权重。"""

    def __init__(self, input_dim: int, n_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, n_experts),
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class MoEPredictor(nn.Module):
    """MoE 气动代理模型。

    输入: (batch, 47) — [RPM, WIND, ANGLE, chord_0..21, twist_0..21]
    输出: (batch, 8) — [μ_T, μ_H, μ_My, μ_Q, σ_T, σ_H, σ_My, σ_Q]

    内部结构:
        SharedEncoder → GatingNetwork (π_k)
                     → K × ExpertNetwork (μ_k, σ_k)
        混合: μ = Σ π_k μ_k
              σ² = Σ π_k (σ_k² + μ_k²) - μ²
    """

    def __init__(self, input_dim: int = 47, output_dim: int = 4,
                 n_experts: int = 4, shared_dim: int = 64,
                 hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_experts = n_experts

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
        )

        self.gating = GatingNetwork(shared_dim, n_experts)

        self.experts = nn.ModuleList([
            ExpertNetwork(shared_dim, hidden_dim, output_dim)
            for _ in range(n_experts)
        ])

    def forward(self, x):
        """前向传播。

        Returns:
            mu: (batch, output_dim) — 混合均值
            sigma: (batch, output_dim) — 混合标准差 (>0)
            gate_weights: (batch, n_experts) — 门控权重
        """
        h = self.shared_encoder(x)
        gate_weights = self.gating(h)  # (batch, K)

        expert_mus = []
        expert_sigmas_sq = []

        for expert in self.experts:
            mu_k, log_sigma_k = expert(h)
            sigma_k_sq = torch.exp(2.0 * log_sigma_k)
            expert_mus.append(mu_k)
            expert_sigmas_sq.append(sigma_k_sq)

        # Stack: (batch, K, output_dim)
        expert_mus = torch.stack(expert_mus, dim=1)
        expert_sigmas_sq = torch.stack(expert_sigmas_sq, dim=1)

        # 门控权重扩展: (batch, K, 1)
        pi = gate_weights.unsqueeze(-1)

        # 混合均值: μ = Σ π_k μ_k
        mu = (pi * expert_mus).sum(dim=1)

        # 混合方差: σ² = Σ π_k (σ_k² + μ_k²) - μ²
        variance = (pi * (expert_sigmas_sq + expert_mus**2)).sum(dim=1) - mu**2
        variance = torch.clamp(variance, min=1e-8)
        sigma = torch.sqrt(variance)

        return mu, sigma, gate_weights

    def predict(self, x: np.ndarray) -> tuple:
        """NumPy 接口: 输入 (N, 47) → 输出 (μ, σ) 各 (N, 4)。"""
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32)
            if x_t.dim() == 1:
                x_t = x_t.unsqueeze(0)
            mu, sigma, _ = self.forward(x_t)
        return mu.numpy(), sigma.numpy()

    def predict_single(self, x: np.ndarray) -> tuple:
        """单样本预测: 输入 (47,) → (μ(4,), σ(4,))。"""
        mu, sigma = self.predict(x.reshape(1, -1))
        return mu[0], sigma[0]


class MoELoss(nn.Module):
    """MoE 训练损失: NLL + balance + smoothness。

    L = L_NLL + λ_bal * L_balance + λ_smooth * L_smooth

    L_NLL = 0.5 * Σ_j [(y_j - μ_j)² / σ_j² + log(σ_j²)]
    L_balance = Var(expert_usage) — 防止某个 expert 独占
    L_smooth = 门控权重的熵 — 鼓励平滑分配
    """

    def __init__(self, balance_weight: float = 0.1, smooth_weight: float = 0.01):
        super().__init__()
        self.balance_weight = balance_weight
        self.smooth_weight = smooth_weight

    def forward(self, mu, sigma, gate_weights, y_true):
        # 异方差 NLL
        variance = sigma**2
        nll = 0.5 * ((y_true - mu)**2 / variance + torch.log(variance))
        nll_loss = nll.mean()

        # Balance loss: 各 expert 平均使用率的方差
        avg_gate = gate_weights.mean(dim=0)  # (K,)
        balance_loss = torch.var(avg_gate) * gate_weights.shape[1]

        # Smoothness loss: 负熵 (鼓励门控不要太尖锐)
        entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=-1).mean()
        smooth_loss = -entropy

        total = nll_loss + self.balance_weight * balance_loss + self.smooth_weight * smooth_loss

        return total, {
            "nll": nll_loss.item(),
            "balance": balance_loss.item(),
            "smooth": smooth_loss.item(),
            "total": total.item(),
        }
