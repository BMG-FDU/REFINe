import os, json, time, copy, random, math, itertools
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from shapely.geometry import LineString, Point
"""
Baseline SAC optimizer for parametric welded network design.

This script couples a parametric graph generator, a surrogate load predictor,
and a Soft Actor-Critic search loop. The optimization explores geometric
offset parameters that improve predicted load response while controlling
material usage through total fiber length.
"""
from features.load_predictor import LoadPredictor
from features.graph_feature_extractor import GraphFeatureExtractor

# Centralized runtime configuration.
# Values are defined as class attributes so that experiment scripts can patch
# them before launching a run.
class RunConfig:
    model_dir        = None

    base_mode        = None
    base             = None
    base_random_dim  = None
    base_random_low  = None
    base_random_high = None

    sac_mode         = None
    horizon          = None

    n_runs           = None
    ep_max           = None
    seed_base        = None

    gb_s             = None
    gb_k             = None
    gb_t             = None
    gb_sc            = None

    act_bound        = None
    p_low            = None
    p_high           = None
    alpha            = None
    beta             = None
    bonus_both       = None
    penalty_l        = None
    l_thresh         = None

    lr               = None
    gamma            = None
    tau              = None
    alpha_init       = None
    auto_alpha       = None
    batch_size       = None
    warmup           = None
    utd              = None
    buf_cap          = None
    hidden_dim       = None
    # Validate that all required experiment parameters have been assigned.
    # The checks also enforce consistency between base initialization mode and
    # SAC rollout mode.
    @classmethod
    def validate(cls):
        required = [
            "model_dir", "base_mode", "sac_mode",
            "n_runs", "ep_max", "seed_base",
            "gb_s", "gb_k", "gb_t", "gb_sc",
            "act_bound", "p_low", "p_high",
            "alpha", "beta", "bonus_both", "penalty_l", "l_thresh",
            "lr", "gamma", "tau", "alpha_init", "auto_alpha",
            "batch_size", "warmup", "utd", "buf_cap", "hidden_dim",
        ]
        missing = [f for f in required if getattr(cls, f) is None]
        if missing:
            raise ValueError("RunConfig fields not set: " + ", ".join(missing))
        if cls.base_mode == "provided":
            if cls.base is None:
                raise ValueError("RunConfig.base required when base_mode='provided'.")
        elif cls.base_mode == "random":
            for f in ["base_random_dim", "base_random_low", "base_random_high"]:
                if getattr(cls, f) is None:
                    raise ValueError(f"RunConfig.{f} required when base_mode='random'.")
        else:
            raise ValueError("base_mode must be 'provided' or 'random'.")
        if cls.sac_mode == "multi_step":
            if cls.horizon is None:
                raise ValueError("RunConfig.horizon required when sac_mode='multi_step'.")
        elif cls.sac_mode != "single_step":
            raise ValueError("sac_mode must be 'single_step' or 'multi_step'.")

# Resolve the initial geometric parameter vector.
# A provided vector enables deterministic comparison, while random initialization
# supports repeated trials with controlled seeds.
def _resolve_base(cfg, seed):
    if cfg.base_mode == "provided":
        return list(cfg.base)
    rng = np.random.RandomState(seed)
    return list(rng.uniform(cfg.base_random_low,
                            cfg.base_random_high,
                            cfg.base_random_dim).astype(np.float32))

# Compute graph edge-length statistics used by the reward and progress reports.
# Keeping this utility separate from the environment makes the reward logic
# easier to adjust.
def edge_lengths(G):
    pos = nx.get_node_attributes(G, 'pos')
    l = [math.hypot(*(np.subtract(pos[u], pos[v]))) for u, v in G.edges()]
    if not l:
        return 0., 0., 0.
    a = np.asarray(l, float)
    return float(a.mean()), float(a.std()), float(a.sum())


def total_edge_length(G):
    return edge_lengths(G)[2]


def moving_average(a, w=50):
    o = np.empty_like(a, float)
    for i in range(len(a)):
        o[i] = a[max(0, i - w + 1):i + 1].mean()
    return o

# GraphBuilder defines the parametric geometry family used by the optimizer.
# It separates cell construction, symmetric perturbation, tiling, and topological
# cleanup so that each stage can be modified independently.
class GraphBuilder:
    def __init__(self, s, k, t, sc):
        self.s, self.k, self.t, self.sc = s, k, t, sc
    # Construct the unperturbed square base cell.
    # Intermediate points on each side define the controllable geometric degrees
    # of freedom.
    def base_square(self):
        G = nx.Graph()
        v = {'A': (0, 0), 'B': (self.s, 0), 'C': (self.s, self.s), 'D': (0, self.s)}
        for n, p in v.items():
            G.add_node(n, pos=p)
        for a, b in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]:
            G.add_edge(a, b)
            pa, pb = np.array(v[a]), np.array(v[b])
            for j in range(1, self.k + 1):
                t = j / (self.k + 1)
                p = (1 - t) * pa + t * pb
                n = f"{a}{b}{j}"
                G.add_node(n, pos=(float(p[0]), float(p[1])))
                G.add_edge(a if j == 1 else f"{a}{b}{j - 1}", n)
                if j == self.k:
                    G.add_edge(n, b)
        return G
    # Apply rotationally symmetric offsets to corresponding boundary points.
    # This preserves the intended design symmetry while still allowing nontrivial
    # local geometric variation.
    def apply_offsets(self, G, o):
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        sh = {}
        for i in range(1, self.k + 1):
            dx, dy = o[2 * (i - 1)], o[2 * (i - 1) + 1]
            dx *= self.s
            dy *= self.s
            sh[f'AB{i}'] = (dx, dy)
            sh[f'BC{i}'] = (-dy, dx)
            sh[f'CD{i}'] = (-dx, -dy)
            sh[f'DA{i}'] = (dy, -dx)
        for n, (ox, oy) in sh.items():
            if n in H.nodes:
                x, y = H.nodes[n]['pos']
                H.nodes[n]['pos'] = (x + ox, y + oy)
        H.remove_edges_from(list(H.edges()))
        ns = sorted(H.nodes())
        for i in range(len(ns)):
            H.add_edge(ns[i], ns[(i + 1) % len(ns)])
        return H
    # Tile and scale the perturbed unit cell into a full network specimen.
    # The scaling stage maps abstract cell coordinates into a normalized design size.
    def tile_scale(self, G):
        H = nx.Graph()
        p = nx.get_node_attributes(G, 'pos')
        sf = self.sc / (self.s * self.t)
        for i in range(self.t):
            for j in range(self.t):
                for n, (x, y) in p.items():
                    H.add_node(f"{n}_{i}_{j}",
                               pos=((x + i * self.s) * sf, (y + j * self.s) * sf))
        for i in range(self.t):
            for j in range(self.t):
                for u, v in G.edges():
                    H.add_edge(f"{u}_{i}_{j}", f"{v}_{i}_{j}")
        return H
    # Promote geometric crossings into explicit graph nodes.
    # This makes downstream graph features and load prediction depend on welded
    # topology rather than only visual intersections.
    def weld(self, G):
        H = copy.deepcopy(G)
        e = list(H.edges())
        b = {}
        for (u1, v1), (u2, v2) in itertools.combinations(e, 2):
            p1, p2 = H.nodes[u1]['pos'], H.nodes[v1]['pos']
            q1, q2 = H.nodes[u2]['pos'], H.nodes[v2]['pos']
            L1, L2 = LineString([p1, p2]), LineString([q1, q2])
            if L1.intersects(L2):
                pt = L1.intersection(L2)
                if pt.geom_type == "Point":
                    x, y = float(pt.x), float(pt.y)
                    n = f"IX_{x:.6f}_{y:.6f}"
                    H.add_node(n, pos=(x, y))
                    b.setdefault((u1, v1), []).append((x, y))
                    b.setdefault((u2, v2), []).append((x, y))
        ne = []
        for u, v in e:
            pts = b.get((u, v), [])
            if not pts:
                ne.append((u, v))
                continue
            P, Q = H.nodes[u]['pos'], H.nodes[v]['pos']
            L = LineString([P, Q])
            pv = u
            for x, y in sorted(pts, key=lambda t: L.project(Point(t))):
                n = f"IX_{x:.6f}_{y:.6f}"
                ne.append((pv, n))
                pv = n
            ne.append((pv, v))
        H.remove_edges_from(e)
        H.add_edges_from(ne)
        return H
    # Remove degenerate edges introduced by coincident points or intersection handling.
    # This cleanup avoids zero-length segments in later graph and geometry operations.
    def clean(self, G):
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        for u, v in G.edges():
            if G.nodes[u]['pos'] != G.nodes[v]['pos']:
                H.add_edge(u, v)
        return H
    # Build the final graph through all geometry and topology-processing stages.
    def build(self, o):
        g = self.base_square()
        g = self.apply_offsets(g, o)
        g = self.tile_scale(g)
        g = self.weld(g)
        g = self.clean(g)
        return g

# WeldEnv provides a lightweight reinforcement-learning environment.
# The state is the current parameter vector, and the action represents a bounded
# geometric perturbation.
class WeldEnv:

    def __init__(self,
                 builder,
                 predictor,
                 base,
                 act_bound,
                 p_low,
                 p_high,
                 alpha,
                 beta,
                 bonus_both,
                 penalty_l,
                 l_thresh,
                 step_anchor="base"):
        self.builder = builder
        self.predictor = predictor
        self.base = np.array(base, dtype=np.float32)
        self.dim = len(base)
        self.act_bound = act_bound
        self.p_low, self.p_high = p_low, p_high
        self.alpha, self.beta = alpha, beta
        self.bonus_both = bonus_both
        self.penalty_l = penalty_l
        self.l_thresh = l_thresh
        self.step_anchor = step_anchor

        G0 = builder.build(self.base)
        self.hb = float(predictor.predict_from_graph(G0))
        self.l0 = float(total_edge_length(G0))
        self.g, self.p = G0, self.base.copy()

    @property
    def obs_dim(self):
        return self.dim

    @property
    def act_dim(self):
        return self.dim

    # Reset the environment to the baseline design for a new episode.
    def reset(self):
        self.p = self.base.copy()
        self.g = self.builder.build(self.p)
        return self.p.copy()

    # Apply an action, rebuild the graph, evaluate the surrogate predictor,
    # and convert the task objectives into a scalar reward.
    # The reward combines predicted load improvement and length reduction.
    # A bonus is assigned when both objectives improve relative to the baseline.
    def step(self, action):
        a = action * self.act_bound
        anchor = self.base if self.step_anchor == "base" else self.p
        self.p = np.clip(anchor + a, self.p_low,
                         self.p_high).astype(np.float32)
        self.g = self.builder.build(self.p)

        load = float(self.predictor.predict_from_graph(self.g))
        length = float(total_edge_length(self.g))

        dw = (load - self.hb) / max(abs(self.hb), 1e-6)
        dl = (self.l0 - length) / max(abs(self.l0), 1e-6)

        r = self.alpha * dw + self.beta * dl
        both = bool(dw > 0 and dl > 0)
        if both:
            r += self.bonus_both
        if dl < -self.l_thresh:
            r -= self.penalty_l

        info = {"w": load, "l": length, "dw": dw, "dl": dl, "both": both}
        return self.p.copy(), float(r), True, info

# ReplayBuffer stores off-policy transitions for SAC updates.
class ReplayBuffer:
    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.buf.append((s.copy(), a.copy(), r, s2.copy(), d))

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s, a, r, s2, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.FloatTensor(np.array(a)),
                torch.FloatTensor(np.array(r)).unsqueeze(1),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(np.array(d)).unsqueeze(1))

    def __len__(self):
        return len(self.buf)

# Actor network for the stochastic policy.
# The tanh-squashed Gaussian output keeps sampled actions within a bounded range.
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hid, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(obs_dim, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.mu  = nn.Linear(hid, act_dim)
        self.log_std = nn.Linear(hid, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = (dist.log_prob(z)
                    - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob

# The twin-critic design reduces overestimation bias during policy optimization.
# Soft Actor-Critic implementation used for continuous geometric search.
# The class owns the actor, critic, target critic, entropy coefficient, and
# replay buffer.
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hid):
        super().__init__()
        dim_in = obs_dim + act_dim
        self.q1 = nn.Sequential(nn.Linear(dim_in, hid), nn.ReLU(),
                                nn.Linear(hid, hid),    nn.ReLU(),
                                nn.Linear(hid, 1))
        self.q2 = nn.Sequential(nn.Linear(dim_in, hid), nn.ReLU(),
                                nn.Linear(hid, hid),    nn.ReLU(),
                                nn.Linear(hid, 1))

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


class SAC:
    def __init__(self, obs_dim, act_dim,
                 lr, gamma, tau, alpha_init, auto_alpha,
                 batch_size, warmup, utd, buf_cap, hidden_dim):
        self.gamma, self.tau = gamma, tau
        self.bs, self.warmup = batch_size, warmup
        self.utd = utd

        self.actor  = Actor(obs_dim, act_dim, hid=hidden_dim)
        self.critic = Critic(obs_dim, act_dim, hid=hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(alpha_init), requires_grad=True)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha_init

        self.rb = ReplayBuffer(buf_cap)
    # Select an action from the current policy.
    # Deterministic mode is reserved for evaluation, while stochastic mode is used
    # during training.
    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            if deterministic:
                mu, _ = self.actor(obs_t)
                return torch.tanh(mu).squeeze(0).numpy()
            a, _ = self.actor.sample(obs_t)
            return a.squeeze(0).numpy()
    # Perform one SAC gradient update using a sampled replay batch.
    def _update_once(self):
        s, a, r, s2, d = self.rb.sample(self.bs)

        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1t, q2t = self.critic_target(s2, a2)
            qt = torch.min(q1t, q2t) - self.alpha * logp2
            target = r + self.gamma * (1.0 - d) * qt

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a_new, logp_new = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, a_new)
        actor_loss = (self.alpha * logp_new - torch.min(q1_new, q2_new)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha.exp()
                           * (logp_new.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        for p, pt in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
    # Run pending SAC updates once the replay buffer contains enough warm-up samples.
    def update(self):
        if len(self.rb) < self.warmup:
            return
        for _ in range(self.utd):
            self._update_once()

# Logger stores episode-level progress and tracks the best observed designs.
# Only compact records are kept here to avoid coupling logging to graph objects.
class Logger:
    def __init__(self):
        self.best_r = -float("inf")
        self.best_info = None
        self.best_both_r = -float("inf")
        self.best_both_info = None
        self.n_both = 0
        self.history = []

    def add(self, ep, r, w, l, params, both=False):
        rec = {"ep": ep, "r": float(r), "w": float(w), "l": float(l),
               "both": both,
               "p": params.tolist() if hasattr(params, "tolist") else list(params)}
        self.history.append(rec)

        if both:
            self.n_both += 1

        new_best      = r > self.best_r
        new_both_best = both and r > self.best_both_r

        if new_best:
            self.best_r    = r
            self.best_info = rec.copy()
        if new_both_best:
            self.best_both_r    = r
            self.best_both_info = rec.copy()

        if new_best and new_both_best:
            print(f"  * new best +BOTH  r={r:.4f}  w={w:.2f}  l={l:.2f}")
        elif new_best:
            print(f"  * new best        r={r:.4f}  w={w:.2f}  l={l:.2f}")
        elif new_both_best:
            print(f"  + new both-best   r={r:.4f}  w={w:.2f}  l={l:.2f}")

# Execute a single-step episode in which one action directly proposes a design.
def _episode_single_step(env, sac, log, ep):
    s = env.reset()
    a = sac.select_action(s)
    s2, r, done, info = env.step(a)
    sac.rb.push(s, a, r, s2, float(done))
    sac.update()
    log.add(ep, r, info["w"], info["l"], env.p, both=info["both"])

# Execute a multi-step episode in which actions iteratively refine the design.
def _episode_multi_step(env, sac, log, ep, horizon):
    s = env.reset()
    total_r   = 0.0
    last_info = None
    for h in range(horizon):
        a         = sac.select_action(s)
        done_flag = float(h == horizon - 1)
        s2, r, _, info = env.step(a)
        sac.rb.push(s, a, r, s2, done_flag)
        sac.update()
        total_r   += r
        last_info  = info
        s          = s2
    log.add(ep, total_r, last_info["w"], last_info["l"], env.p,
            both=last_info["both"])

# Train one independent SAC run with a fixed random seed.
# The run includes graph construction, surrogate predictor setup, environment
# creation, optimization, and progress reporting.
def train(run_id, seed):
    cfg = RunConfig
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    extractor = GraphFeatureExtractor()
    predictor = LoadPredictor(model_dir=cfg.model_dir, extractor=extractor)
    builder   = GraphBuilder(s=cfg.gb_s, k=cfg.gb_k, t=cfg.gb_t, sc=cfg.gb_sc)
    base      = _resolve_base(cfg, seed)

    step_anchor = "base" if cfg.sac_mode == "single_step" else "current"
    env = WeldEnv(
        builder, predictor, base,
        act_bound=cfg.act_bound,
        p_low=cfg.p_low,        p_high=cfg.p_high,
        alpha=cfg.alpha,        beta=cfg.beta,
        bonus_both=cfg.bonus_both,
        penalty_l=cfg.penalty_l,
        l_thresh=cfg.l_thresh,
        step_anchor=step_anchor,
    )

    print(f"[run {run_id}] base = [{', '.join(f'{v:+.2f}' for v in base)}]")
    print(f"[run {run_id}] w0={env.hb:.2f}  l0={env.l0:.2f}  mode={cfg.sac_mode}")

    sac = SAC(
        obs_dim=env.obs_dim, act_dim=env.act_dim,
        lr=cfg.lr,            gamma=cfg.gamma,
        tau=cfg.tau,          alpha_init=cfg.alpha_init,
        auto_alpha=cfg.auto_alpha,
        batch_size=cfg.batch_size, warmup=cfg.warmup,
        utd=cfg.utd,          buf_cap=cfg.buf_cap,
        hidden_dim=cfg.hidden_dim,
    )
    log = Logger()

    for ep in range(1, cfg.ep_max + 1):
        if cfg.sac_mode == "single_step":
            _episode_single_step(env, sac, log, ep)
        else:
            _episode_multi_step(env, sac, log, ep, cfg.horizon)

        if ep % 200 == 0:
            last = log.history[-1]
            print(f"  [{ep:05d}/{cfg.ep_max}] r={last['r']:+.4f}  "
                  f"w={last['w']:.2f}  l={last['l']:.2f}  "
                  f"best={log.best_r:.4f}  "
                  f"both_rate={log.n_both/ep:.1%}  "
                  f"alpha={sac.alpha:.4f}")

    return log

# Main experiment driver for repeated runs.
# Multiple seeds are used to estimate robustness and to select a recommended
# design from successful trials.
def main():
    RunConfig.validate()
    cfg = RunConfig
    sep = "=" * 90

    print(sep)
    print(f"  SAC  {cfg.n_runs} runs x {cfg.ep_max} eps"
          f"  sac_mode={cfg.sac_mode}  base_mode={cfg.base_mode}")
    print(sep)

    all_logs = []
    for i in range(cfg.n_runs):
        seed = i * 137 + cfg.seed_base
        print(f"\n{'-'*90}")
        print(f"  Run {i+1}/{cfg.n_runs}   seed={seed}")
        print(f"{'-'*90}")

        log = train(run_id=i + 1, seed=seed)
        all_logs.append(log)

        b = log.best_info
        bb = log.best_both_info
        print(f"\n  Run {i+1} done:")
        print(
            f"    best:      r={b['r']:+.4f}  w={b['w']:.2f}  l={b['l']:.2f}  (ep {b['ep']})"
        )
        if bb:
            print(
                f"    both-best: r={bb['r']:+.4f}  w={bb['w']:.2f}  l={bb['l']:.2f}  (ep {bb['ep']})"
            )
        else:
            print(f"    both-best: not found")
        print(f"    both_rate: {log.n_both/cfg.ep_max:.1%}")

    print(f"\n\n{sep}")
    print("  Summary")
    print(sep)
    print(f" Run | {'best_r':>9} {'best_w':>8} {'best_l':>8} | "
          f"{'both_r':>9} {'both_w':>8} {'both_l':>8} | {'rate':>6}")
    print("-" * 90)

    for i, log in enumerate(all_logs):
        b = log.best_info
        bb = log.best_both_info
        br = f"{bb['r']:9.4f}" if bb else "      N/A"
        bw = f"{bb['w']:8.2f}" if bb else "     N/A"
        bl = f"{bb['l']:8.2f}" if bb else "     N/A"
        print(f"  {i+1:2d} | {b['r']:9.4f} {b['w']:8.2f} {b['l']:8.2f} | "
              f"{br} {bw} {bl} | {log.n_both/cfg.ep_max:6.1%}")

    brs = [log.best_r for log in all_logs]
    bws = [log.best_info['w'] for log in all_logs]
    print(f"\n  best_r : {np.mean(brs):.4f} +/- {np.std(brs):.4f}"
          f"  (range {np.min(brs):.4f} ~ {np.max(brs):.4f})")
    print(f"  best_w : {np.mean(bws):.2f} +/- {np.std(bws):.2f}")

    valid = [(i, log) for i, log in enumerate(all_logs)
             if log.best_both_info is not None]
    if valid:
        vrs = [log.best_both_r for _, log in valid]
        print(f"  both_r : {np.mean(vrs):.4f} +/- {np.std(vrs):.4f}"
              f"  ({len(valid)}/{cfg.n_runs} runs found)")

    if valid:
        ci, clog = max(valid, key=lambda x: x[1].best_both_r)
        ch = clog.best_both_info
        base_arr = np.array(_resolve_base(cfg, ci * 137 + cfg.seed_base))
        p = np.array(ch['p'])
        delta = p - base_arr
        print(f"\n  * recommended  (Run {ci+1}, both-best)")
        print(f"    reward = {ch['r']:+.4f}")
        print(f"    w      = {ch['w']:.2f}")
        print(f"    l      = {ch['l']:.2f}")
        print(f"    delta  = [{', '.join(f'{d:+.4f}' for d in delta)}]")
        print(f"    params = [{', '.join(f'{v:+.4f}' for v in p)}]")
    else:
        best_idx = int(np.argmax(brs))
        b = all_logs[best_idx].best_info
        print(
            f"\n  no both-solution found. global best from Run {best_idx+1}:")
        print(f"    w={b['w']:.2f}  l={b['l']:.2f}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "n_runs": cfg.n_runs,
        "ep_max": cfg.ep_max,
        "sac_mode": cfg.sac_mode,
        "base_mode": cfg.base_mode,
        "runs": [],
    }
    for i, log in enumerate(all_logs):
        out["runs"].append({
            "run": i + 1,
            "seed": i * 137 + cfg.seed_base,
            "best": log.best_info,
            "best_both": log.best_both_info,
            "both_rate": round(log.n_both / cfg.ep_max, 4),
            "history": log.history,
        })
    path = f"sac_opt_{ts}.json"
    # Save the aggregated experiment history to JSON for later inspection,
    # plotting, or comparison with other optimization settings.
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved -> {path}")
    print(sep)


if __name__ == "__main__":
    main()
