"""SAC with a LayerNorm critic ensemble and RLPD-style mixed sampling (HIL-SERL-style learner).

Uncertainty of a state is the standard deviation across critics of Q(s, a*), with a* the deterministic policy action.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uhitl.envs import ACT_DIM, ACT_HIGH, ACT_LOW, OBS_DIM


def mlp(inp, out, hidden=256, layer_norm=False, dropout=0.0):
    layers = []
    d = inp
    for _ in range(2):
        layers.append(nn.Linear(d, hidden))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(OBS_DIM, 2 * ACT_DIM)
        self.register_buffer("scale", torch.as_tensor((ACT_HIGH - ACT_LOW) / 2))
        self.register_buffer("bias", torch.as_tensor((ACT_HIGH + ACT_LOW) / 2))

    def forward(self, obs, deterministic=False):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        if deterministic:
            u = mu
        else:
            u = mu + log_std.exp() * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (
            -0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        ).sum(-1)
        logp = logp - torch.log(self.scale * (1 - a.pow(2)) + 1e-6).sum(-1)
        return a * self.scale + self.bias, logp


class EnsembleLinear(nn.Module):
    """K independent linear layers evaluated in one batched matrix multiply."""

    def __init__(self, k, inp, out):
        super().__init__()
        bound = 1 / np.sqrt(inp)
        self.w = nn.Parameter(torch.empty(k, inp, out).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(k, 1, out).uniform_(-bound, bound))

    def forward(self, x):  # x: (K, B, inp)
        return torch.baddbmm(self.b, x, self.w)


class CriticEnsemble(nn.Module):
    """K critics (Linear-[Dropout]-LayerNorm-ReLU x2 - Linear), vectorised over the ensemble dimension."""

    def __init__(self, k, dropout=0.0, hidden=256):
        super().__init__()
        self.k, self.dropout = k, dropout
        self.l1 = EnsembleLinear(k, OBS_DIM + ACT_DIM, hidden)
        self.l2 = EnsembleLinear(k, hidden, hidden)
        self.l3 = EnsembleLinear(k, hidden, 1)
        self.g1, self.b1 = nn.Parameter(torch.ones(k, 1, hidden)), nn.Parameter(
            torch.zeros(k, 1, hidden)
        )
        self.g2, self.b2 = nn.Parameter(torch.ones(k, 1, hidden)), nn.Parameter(
            torch.zeros(k, 1, hidden)
        )

    def _hidden(self, h, layer, gain, bias, force_dropout):
        h = layer(h)
        if self.dropout > 0:
            h = F.dropout(h, self.dropout, training=self.training or force_dropout)
        h = F.layer_norm(h, h.shape[-1:]) * gain + bias  # per-member LayerNorm affine
        return F.relu(h)

    def forward(self, obs, act, force_dropout=False):
        x = torch.cat([obs, act], dim=-1).unsqueeze(0).expand(self.k, -1, -1)
        h = self._hidden(x, self.l1, self.g1, self.b1, force_dropout)
        h = self._hidden(h, self.l2, self.g2, self.b2, force_dropout)
        return self.l3(h).squeeze(-1)  # (K, B)


class Buffer:
    def __init__(self, size):
        self.o = np.zeros((size, OBS_DIM), np.float32)
        self.a = np.zeros((size, ACT_DIM), np.float32)
        self.r = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, OBS_DIM), np.float32)
        self.d = np.zeros(size, np.float32)
        self.n, self.i, self.size = 0, 0, size

    def add(self, o, a, r, o2, d):
        (
            self.o[self.i],
            self.a[self.i],
            self.r[self.i],
            self.o2[self.i],
            self.d[self.i],
        ) = (o, a, r, o2, d)
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, rng, n):
        idx = rng.integers(0, self.n, n)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]


class SACAgent:
    def __init__(
        self,
        num_critics=5,
        num_target_min=2,
        utd=4,
        batch=256,
        gamma=0.97,
        lr=3e-4,
        tau=0.005,
        critic_dropout=0.0,
        init_temperature=0.01,
        bc_weight=0.0,
        device="cpu",
        seed=0,
    ):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.k, self.m, self.utd, self.batch, self.gamma, self.tau = (
            num_critics,
            num_target_min,
            utd,
            batch,
            gamma,
            tau,
        )
        self.actor = Actor().to(self.device)
        self.critic = CriticEnsemble(num_critics, critic_dropout).to(self.device)
        self.target = CriticEnsemble(num_critics, critic_dropout).to(self.device)
        self.target.load_state_dict(self.critic.state_dict())
        # a small initial temperature keeps the entropy bonus from swamping the sparse success reward
        self.log_alpha = torch.full(
            (1,),
            float(np.log(init_temperature)),
            requires_grad=True,
            device=self.device,
        )
        self.target_entropy = -float(ACT_DIM)
        self.bc_weight = bc_weight  # auxiliary BC loss on intervention samples (0 = pure SAC)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_t = torch.optim.Adam([self.log_alpha], lr=lr)
        self.online = Buffer(200_000)
        self.interv = Buffer(200_000)

    def _t(self, x):
        return torch.as_tensor(x, device=self.device)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _ = self.actor(self._t(obs[None]), deterministic)
        return a[0].cpu().numpy()

    @torch.no_grad()
    def q_values(self, obs, act=None):
        """Per-critic Q(s, a) for one state; a defaults to the deterministic policy action."""
        o = self._t(obs[None])
        a = self.actor(o, deterministic=True)[0] if act is None else self._t(act[None])
        return self.critic(o, a)[:, 0].cpu().numpy()

    @torch.no_grad()
    def mc_dropout_q(self, obs, samples=10):
        """Q(s, a*) of the first critic under `samples` stochastic dropout masks."""
        o = self._t(obs[None]).expand(samples, -1)
        a = self.actor(o, deterministic=True)[0]
        return self.critic(o, a, force_dropout=True)[0].cpu().numpy()

    def _sample(self):
        if self.interv.n > 0 and self.online.n > 0:
            half = self.batch // 2
            parts = [
                self.online.sample(self.rng, half),
                self.interv.sample(self.rng, self.batch - half),
            ]
            return [
                self._t(np.concatenate([parts[0][j], parts[1][j]])) for j in range(5)
            ]
        buf = self.online if self.online.n > 0 else self.interv
        return [self._t(x) for x in buf.sample(self.rng, self.batch)]

    def update(self):
        if self.online.n + self.interv.n < self.batch:
            return {}
        alpha = self.log_alpha.exp().detach()
        for _ in range(self.utd):
            o, a, r, o2, d = self._sample()
            with torch.no_grad():
                a2, logp2 = self.actor(o2)
                tq = self.target(o2, a2)
                idx = torch.as_tensor(
                    self.rng.choice(self.k, self.m, replace=False), device=self.device
                )
                tq = tq[idx].min(0).values - alpha * logp2
                y = r + self.gamma * (1 - d) * tq
            q = self.critic(o, a)
            loss_c = F.mse_loss(q, y.expand_as(q))
            self.opt_c.zero_grad()
            loss_c.backward()
            self.opt_c.step()
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.target.parameters()):
                    tp.mul_(1 - self.tau).add_(self.tau * p)
        a_pi, logp = self.actor(o)
        loss_a = (alpha * logp - self.critic(o, a_pi).mean(0)).mean()
        if self.bc_weight > 0 and self.interv.n > 0 and self.online.n > 0:
            half = self.batch // 2  # second half of the batch comes from the intervention buffer
            det, _ = self.actor(o[half:], deterministic=True)
            loss_a = loss_a + self.bc_weight * F.mse_loss(det, a[half:])
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        loss_t = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_t.zero_grad()
        loss_t.backward()
        self.opt_t.step()
        return {"loss_c": loss_c.item(), "loss_a": loss_a.item(), "alpha": alpha.item()}

    def update_bc(self):
        """HG-DAgger-style learner: regress the deterministic actor onto intervention actions only."""
        o, a, _, _, _ = [self._t(x) for x in self.interv.sample(self.rng, self.batch)]
        pred, _ = self.actor(o, deterministic=True)
        loss = F.mse_loss(pred, a)
        self.opt_a.zero_grad()
        loss.backward()
        self.opt_a.step()
        return {"loss_bc": loss.item()}

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.target.load_state_dict(sd["target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
