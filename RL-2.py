import os, json, time, copy, random, math, itertools, csv
"""
Extended SAC optimizer for constrained welded network design.

This version expands the baseline optimizer with configurable reward families,
state representations, constraint handling, CSV logging, and multi-run
experiment management. It is intended for systematic comparison of optimization
settings under surrogate-model evaluation.
"""
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from shapely.geometry import LineString, Point
from tqdm.auto import tqdm

from features.load_predictor import LoadPredictor
from features.graph_feature_extractor import GraphFeatureExtractor

# Central experiment configuration.
# The fields cover data paths, geometric parameterization, reward construction,
# SAC hyperparameters, logging behavior.
class RunConfig:

    model_dir  = None
    output_dir = None

    base_mode        = None
    base             = None
    base_random_dim  = None
    base_random_low  = None
    base_random_high = None

    reward_mode      = None
    constrained_type = None

    state_mode      = None
    sac_mode        = None
    horizon         = None
    constraint_mode = None

    n_runs    = None
    ep_max    = None
    seed_base = None
    debug     = None

    gb_s  = None
    gb_k  = None
    gb_t  = None
    gb_sc = None

    act_bound = None
    p_low     = None
    p_high    = None

    lw_lambda_w = None
    lw_hb       = None

    fl_alpha       = None
    fl_beta        = None
    fl_tol_pct     = None
    fl_close_pct   = None
    fl_tight_pct   = None
    fl_bonus_sat   = None
    fl_bonus_close = None
    fl_bonus_tight = None
    fl_bonus_both  = None

    fw_m_target       = None
    fw_delta          = None
    fw_hb             = None
    fw_lambda_penalty = None

    lr         = None
    gamma      = None
    tau        = None
    alpha_init = None
    auto_alpha = None
    batch_size = None
    warmup     = None
    utd        = None
    buf_cap    = None
    hidden_dim = None

    @classmethod
    # Validate the configuration before any expensive computation starts.
    def validate(cls):
        _req = [
            "model_dir", "output_dir",
            "base_mode", "reward_mode", "state_mode",
            "sac_mode", "constraint_mode",
            "n_runs", "ep_max", "seed_base", "debug",
            "gb_s", "gb_k", "gb_t", "gb_sc",
            "act_bound", "p_low", "p_high",
            "lr", "gamma", "tau", "alpha_init", "auto_alpha",
            "batch_size", "warmup", "utd", "buf_cap", "hidden_dim",
        ]
        _miss = [f for f in _req if getattr(cls, f) is None]
        if _miss:
            raise ValueError(", ".join(_miss))
        if cls.base_mode == "provided":
            if cls.base is None:
                raise ValueError("base")
        elif cls.base_mode == "random":
            for _f in ("base_random_dim", "base_random_low", "base_random_high"):
                if getattr(cls, _f) is None:
                    raise ValueError(_f)
        else:
            raise ValueError("base_mode")
        if cls.sac_mode == "multi_step":
            if cls.horizon is None:
                raise ValueError("horizon")
        elif cls.sac_mode != "single_step":
            raise ValueError("sac_mode")
        if cls.constraint_mode not in ("clip", "radial"):
            raise ValueError("constraint_mode")
        if cls.state_mode not in ("simple", "full"):
            raise ValueError("state_mode")
        if cls.reward_mode == "loss_weight":
            for _f in ("lw_lambda_w", "lw_hb"):
                if getattr(cls, _f) is None:
                    raise ValueError(_f)
        elif cls.reward_mode == "constrained":
            if cls.constrained_type not in ("tiered", "quadratic"):
                raise ValueError("constrained_type")
            if cls.constrained_type == "tiered":
                for _f in ("fl_alpha", "fl_beta",
                           "fl_tol_pct", "fl_close_pct", "fl_tight_pct",
                           "fl_bonus_sat", "fl_bonus_close",
                           "fl_bonus_tight", "fl_bonus_both"):
                    if getattr(cls, _f) is None:
                        raise ValueError(_f)
            else:
                for _f in ("fw_m_target", "fw_delta", "fw_hb", "fw_lambda_penalty"):
                    if getattr(cls, _f) is None:
                        raise ValueError(_f)
        else:
            raise ValueError("reward_mode")

    @classmethod
    # Export the configuration as a serializable dictionary for experiment records.
    def as_dict(cls):
        _skip = {"validate", "as_dict"}
        return {k: getattr(cls, k) for k in dir(cls)
                if not k.startswith("_") and k not in _skip
                and not callable(getattr(cls, k))}

# Resolve the baseline design parameters from either a provided vector or a
# seeded random initialization.
def _resolve_base(cfg, seed):
    if cfg.base_mode == "provided":
        return list(cfg.base)
    _rng = np.random.RandomState(seed)
    return list(_rng.uniform(cfg.base_random_low,
                             cfg.base_random_high,
                             cfg.base_random_dim).astype(np.float32))

# Compute total graph edge length as a material-usage proxy.
def _total_edge_length(G):
    _pos = nx.get_node_attributes(G, 'pos')
    return sum(math.hypot(*(np.subtract(_pos[u], _pos[v])))
               for u, v in G.edges())

# Compute a compact graph-level state descriptor.
# These features are intentionally lightweight compared with the full predictor
# features to keep the RL state small.
def _graph_features(G):
    _pos  = nx.get_node_attributes(G, 'pos')
    _nn   = float(G.number_of_nodes())
    _ne   = float(G.number_of_edges())
    _degs = [d for _, d in G.degree()]
    _md   = float(np.mean(_degs)) if _degs else 0.0
    if _ne > 0:
        _lens = np.array([math.hypot(*(np.subtract(_pos[u], _pos[v])))
                          for u, v in G.edges()], dtype=float)
        _mel, _sel, _tel = float(_lens.mean()), float(_lens.std()), float(_lens.sum())
    else:
        _mel = _sel = _tel = 0.0
    return np.array([_nn, _ne, _md, _mel, _sel, _tel], dtype=np.float32)

# Ensure that an output directory exists before writing logs or summaries.
def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

# GraphBuilder defines the constrained parametric design space.
# The geometry pipeline is kept modular so that symmetry rules, tiling density,
# and welding logic can be changed independently.
class GraphBuilder:

    def __init__(self, s, k, t, sc):
        self.s, self.k, self.t, self.sc = s, k, t, sc
    # Create the base square cell with controllable side points.
    def _base_square(self):
        G  = nx.Graph()
        _v = {'A': (0, 0), 'B': (self.s, 0),
              'C': (self.s, self.s), 'D': (0, self.s)}
        for n, p in _v.items():
            G.add_node(n, pos=p)
        for a, b in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]:
            G.add_edge(a, b)
            _pa, _pb = np.array(_v[a]), np.array(_v[b])
            for j in range(1, self.k + 1):
                _frac = j / (self.k + 1)
                _p    = (1 - _frac) * _pa + _frac * _pb
                _nm   = f"{a}{b}{j}"
                G.add_node(_nm, pos=(float(_p[0]), float(_p[1])))
                G.add_edge(a if j == 1 else f"{a}{b}{j-1}", _nm)
                if j == self.k:
                    G.add_edge(_nm, b)
        return G
    # Apply symmetry-preserving offsets to the base-cell control points.
    def _apply_offsets(self, G, o):
        H   = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        _sh = {}
        for i in range(1, self.k + 1):
            _dx = o[2 * (i - 1)] * self.s
            _dy = o[2 * (i - 1) + 1] * self.s
            _sh[f'AB{i}'] = ( _dx,  _dy)
            _sh[f'BC{i}'] = (-_dy,  _dx)
            _sh[f'CD{i}'] = (-_dx, -_dy)
            _sh[f'DA{i}'] = ( _dy, -_dx)
        for n, (ox, oy) in _sh.items():
            if n in H.nodes:
                x, y = H.nodes[n]['pos']
                H.nodes[n]['pos'] = (x + ox, y + oy)
        H.remove_edges_from(list(H.edges()))
        _ns = sorted(H.nodes())
        for i in range(len(_ns)):
            H.add_edge(_ns[i], _ns[(i + 1) % len(_ns)])
        return H
    # Convert the perturbed cell into a tiled and scaled specimen graph.
    def _tile_scale(self, G):
        H   = nx.Graph()
        _p  = nx.get_node_attributes(G, 'pos')
        _sf = self.sc / (self.s * self.t)
        for i in range(self.t):
            for j in range(self.t):
                for n, (x, y) in _p.items():
                    H.add_node(f"{n}_{i}_{j}",
                               pos=((x + i * self.s) * _sf,
                                    (y + j * self.s) * _sf))
        for i in range(self.t):
            for j in range(self.t):
                for u, v in G.edges():
                    H.add_edge(f"{u}_{i}_{j}", f"{v}_{i}_{j}")
        return H
    # Split intersecting segments by inserting explicit intersection nodes.
    # This step aligns the graph topology with the intended welded geometry.
    def _weld(self, G):
        H  = copy.deepcopy(G)
        _e = list(H.edges())
        _b = {}
        for (u1, v1), (u2, v2) in itertools.combinations(_e, 2):
            L1 = LineString([H.nodes[u1]['pos'], H.nodes[v1]['pos']])
            L2 = LineString([H.nodes[u2]['pos'], H.nodes[v2]['pos']])
            if L1.intersects(L2):
                _pt = L1.intersection(L2)
                if _pt.geom_type == "Point":
                    x, y = float(_pt.x), float(_pt.y)
                    _n   = f"IX_{x:.6f}_{y:.6f}"
                    H.add_node(_n, pos=(x, y))
                    _b.setdefault((u1, v1), []).append((x, y))
                    _b.setdefault((u2, v2), []).append((x, y))
        _ne = []
        for u, v in _e:
            _pts = _b.get((u, v), [])
            if not _pts:
                _ne.append((u, v))
                continue
            L  = LineString([H.nodes[u]['pos'], H.nodes[v]['pos']])
            pv = u
            for x, y in sorted(_pts, key=lambda t_: L.project(Point(t_))):
                _n = f"IX_{x:.6f}_{y:.6f}"
                _ne.append((pv, _n))
                pv = _n
            _ne.append((pv, v))
        H.remove_edges_from(_e)
        H.add_edges_from(_ne)
        return H
    # Remove zero-length or self-joining edges after geometric processing.
    def _clean(self, G):
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        for u, v in G.edges():
            if G.nodes[u]['pos'] != G.nodes[v]['pos']:
                H.add_edge(u, v)
        return H
    # Execute the complete graph-generation pipeline for a parameter vector.
    def build(self, o):
        g = self._base_square()
        g = self._apply_offsets(g, o)
        g = self._tile_scale(g)
        g = self._weld(g)
        g = self._clean(g)
        return g

# WeldEnvB implements the optimization task for SAC.
# It supports different state encodings, parameter constraints, and reward modes
# while using the same graph builder and surrogate predictor.
class WeldEnvB:

    def __init__(self, builder, predictor, base, cfg):
        self.builder   = builder
        self.predictor = predictor
        self.base      = np.array(base, dtype=np.float32)
        self.dim       = len(base)
        self.cfg       = cfg

        _G0      = builder.build(self.base)
        self.w0  = float(predictor.predict_from_graph(_G0))
        self.l0  = float(_total_edge_length(_G0))
        self.l_target = self.l0

        if cfg.state_mode == "full":
            self.gf0 = _graph_features(_G0)

        self._obs_dim = (self.dim
                         if cfg.state_mode == "simple"
                         else self.dim * 2 + 8)

        self.g, self.p = _G0, self.base.copy()

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def act_dim(self):
        return self.dim
    # Construct either a simple parameter-only state or a fuller state augmented
    # with boundary distance, graph statistics, and normalized load indicators.
    def _get_state(self, p, G, w=None):
        if self.cfg.state_mode == "simple":
            return p.copy()
        if w is None:
            w = float(self.predictor.predict_from_graph(G))
        _bd  = np.minimum(p - self.cfg.p_low, self.cfg.p_high - p)
        _gf  = _graph_features(G) / (self.gf0 + 1e-6)
        _wn  = w / max(abs(self.w0), 1e-6)
        _dwn = (w - self.w0) / max(abs(self.w0), 1e-6)
        return np.concatenate([p, _bd, _gf, [_wn, _dwn]]).astype(np.float32)
    # Apply the selected feasibility rule to raw parameters.
    # Clipping enforces box constraints, while radial projection restricts the
    # parameter vector within a global norm bound.
    def _constrain(self, p_raw):
        if self.cfg.constraint_mode == "radial":
            _norm = np.linalg.norm(p_raw)
            if _norm > 0.5:
                p_raw = p_raw * 0.5 / _norm
        return np.clip(p_raw, self.cfg.p_low, self.cfg.p_high).astype(np.float32)
    # Reset to the baseline design at the start of an episode.
    def reset(self):
        self.p = self.base.copy()
        self.g = self.builder.build(self.p)
        return self._get_state(self.p, self.g, w=self.w0)
    # Apply a bounded action, rebuild the graph, predict load, evaluate length,
    # and return the next state and task reward.
    def step(self, action):
        _a      = action * self.cfg.act_bound
        _anchor = self.base if self.cfg.sac_mode == "single_step" else self.p
        self.p  = self._constrain(_anchor + _a)
        self.g  = self.builder.build(self.p)
        w       = float(self.predictor.predict_from_graph(self.g))
        l       = float(_total_edge_length(self.g))
        r, info = self._reward(w, l)
        s2      = self._get_state(self.p, self.g, w=w)
        return s2, float(r), True, info
    # Dispatch reward computation according to the selected reward mode.
    def _reward(self, w, l):
        if self.cfg.reward_mode == "loss_weight":
            return self._rew_lw(w, l)
        return self._rew_constrained(w, l)
    # Reward mode balancing load gain against material length.
    # This mode is useful as a compact baseline scalarization.
    def _rew_lw(self, w, l):
        cfg   = self.cfg
        r     = (w / max(abs(self.w0), 1e-6)
                 - cfg.lw_lambda_w * l / max(abs(self.l0), 1e-6))
        _imp  = w > self.w0
        _mat  = l < self.l0
        _both = _imp and _mat
        if _both:
            r += cfg.lw_hb
        _dw  = (w - self.w0) / max(abs(self.w0), 1e-6)
        _gap = abs(l - self.l0) / max(abs(self.l0), 1e-6)
        return r, dict(w=w, l=l, dw=_dw, gap=_gap,
                       sat=_both, close=False, tight=False,
                       imp=_imp, both=_both)

    def _rew_constrained(self, w, l):
        if self.cfg.constrained_type == "tiered":
            return self._rew_tiered(w, l)
        return self._rew_quadratic(w, l)

    def _rew_tiered(self, w, l):
        cfg    = self.cfg
        _dw    = (w - self.w0)           / max(abs(self.w0),      1e-6)
        _gap   = abs(l - self.l_target)  / max(abs(self.l_target), 1e-6)
        _sat   = _gap < cfg.fl_tol_pct
        _close = _gap < cfg.fl_close_pct
        _tight = _gap < cfg.fl_tight_pct
        _imp   = w > self.w0
        _both  = _sat and _imp
        r = cfg.fl_alpha * _dw - cfg.fl_beta * _gap
        if _sat:   r += cfg.fl_bonus_sat
        if _close: r += cfg.fl_bonus_close
        if _tight: r += cfg.fl_bonus_tight
        if _both:  r += cfg.fl_bonus_both
        return r, dict(w=w, l=l, dw=_dw, gap=_gap,
                       sat=_sat, close=_close, tight=_tight,
                       imp=_imp, both=_both)
    # Quadratic constrained reward.
    # This formulation penalizes deviation from a target material length while
    # retaining predicted load as the main performance signal.
    def _rew_quadratic(self, w, l):
        cfg   = self.cfg
        _dev  = abs(l - cfg.fw_m_target)
        _sat  = _dev <= cfg.fw_delta
        r     = ((cfg.fw_hb + w) if _sat
                 else (w - cfg.fw_lambda_penalty * (_dev - cfg.fw_delta) ** 2))
        _dw   = (w - self.w0) / max(abs(self.w0), 1e-6)
        _gap  = _dev / max(abs(cfg.fw_m_target), 1e-6)
        _imp  = w > self.w0
        _both = _sat and _imp
        return r, dict(w=w, l=l, dw=_dw, gap=_gap,
                       sat=_sat, close=False, tight=False,
                       imp=_imp, both=_both)

# ReplayBuffer stores off-policy transitions used by SAC.
class ReplayBuffer:

    def __init__(self, cap):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.buf.append((s.copy(), a.copy(), r, s2.copy(), d))

    def sample(self, bs):
        _batch = random.sample(self.buf, bs)
        s, a, r, s2, d = zip(*_batch)
        return (torch.FloatTensor(np.array(s)),
                torch.FloatTensor(np.array(a)),
                torch.FloatTensor(np.array(r)).unsqueeze(1),
                torch.FloatTensor(np.array(s2)),
                torch.FloatTensor(np.array(d)).unsqueeze(1))

    def __len__(self):
        return len(self.buf)

# Actor network for continuous bounded actions.
# The stochastic policy uses reparameterized Gaussian samples followed by tanh
# squashing.
class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hid, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1     = nn.Linear(obs_dim, hid)
        self.fc2     = nn.Linear(hid, hid)
        self.mu      = nn.Linear(hid, act_dim)
        self.log_std = nn.Linear(hid, act_dim)

    def forward(self, x):
        x        = F.relu(self.fc1(x))
        x        = F.relu(self.fc2(x))
        _mu      = self.mu(x)
        _log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return _mu, _log_std

    def sample(self, obs):
        _mu, _log_std = self.forward(obs)
        _std  = _log_std.exp()
        _dist = torch.distributions.Normal(_mu, _std)
        _z    = _dist.rsample()
        _a    = torch.tanh(_z)
        _lp   = (_dist.log_prob(_z)
                 - torch.log(1 - _a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return _a, _lp

# Twin-critic network used by SAC to stabilize value estimation.
class Critic(nn.Module):

    def __init__(self, obs_dim, act_dim, hid):
        super().__init__()
        d = obs_dim + act_dim
        self.q1 = nn.Sequential(nn.Linear(d, hid), nn.ReLU(),
                                nn.Linear(hid, hid), nn.ReLU(),
                                nn.Linear(hid, 1))
        self.q2 = nn.Sequential(nn.Linear(d, hid), nn.ReLU(),
                                nn.Linear(hid, hid), nn.ReLU(),
                                nn.Linear(hid, 1))

    def forward(self, s, a):
        _x = torch.cat([s, a], dim=-1)
        return self.q1(_x), self.q2(_x)

# Soft Actor-Critic optimizer.
# This implementation manages policy learning, value learning, target-network
# updates, entropy adaptation, and replay sampling.
class SAC:

    def __init__(self, obs_dim, act_dim, *,
                 lr, gamma, tau,
                 alpha_init, auto_alpha,
                 batch_size, warmup, utd, buf_cap, hidden_dim):
        self.gamma, self.tau           = gamma, tau
        self.bs, self.warmup, self.utd = batch_size, warmup, utd

        self.actor         = Actor(obs_dim, act_dim, hid=hidden_dim)
        self.critic        = Critic(obs_dim, act_dim, hid=hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -float(act_dim)
            self.log_alpha      = torch.tensor(np.log(alpha_init), requires_grad=True)
            self.alpha_opt      = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha          = self.log_alpha.exp().item()
        else:
            self.alpha = alpha_init

        self.rb = ReplayBuffer(buf_cap)
    # Select an action for exploration or deterministic evaluation.
    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            _obs_t = torch.FloatTensor(obs).unsqueeze(0)
            if deterministic:
                _mu, _ = self.actor(_obs_t)
                return torch.tanh(_mu).squeeze(0).numpy()
            _a, _ = self.actor.sample(_obs_t)
            return _a.squeeze(0).numpy()

    def _update_once(self):
        s, a, r, s2, d = self.rb.sample(self.bs)
        with torch.no_grad():
            _a2, _lp2   = self.actor.sample(s2)
            _q1t, _q2t  = self.critic_target(s2, _a2)
            _qt         = torch.min(_q1t, _q2t) - self.alpha * _lp2
            _target     = r + self.gamma * (1.0 - d) * _qt
        _q1, _q2 = self.critic(s, a)
        _closs   = F.mse_loss(_q1, _target) + F.mse_loss(_q2, _target)
        self.critic_opt.zero_grad()
        _closs.backward()
        self.critic_opt.step()
        _a_new, _lp_new = self.actor.sample(s)
        _q1n, _q2n      = self.critic(s, _a_new)
        _aloss = (self.alpha * _lp_new - torch.min(_q1n, _q2n)).mean()
        self.actor_opt.zero_grad()
        _aloss.backward()
        self.actor_opt.step()
        if self.auto_alpha:
            _al = -(self.log_alpha.exp()
                    * (_lp_new.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            _al.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()
        for p, pt in zip(self.critic.parameters(),
                         self.critic_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
    # Trigger SAC updates after enough replay samples have been collected.
    def update(self):
        if len(self.rb) < self.warmup:
            return
        for _ in range(self.utd):
            self._update_once()

# LoggerB writes episode-level records and improvement events to CSV files.
# Separate logs make it easier to audit best solutions and constraint-satisfying
# candidates after long runs.
class LoggerB:

    def __init__(self, run_dir, run_id, w0, l0, n_params):
        self.run_dir  = run_dir
        self.run_id   = run_id
        self.w0, self.l0 = w0, l0
        self.n_params = n_params
        self.best_r         = -float("inf")
        self.best_info      = None
        self.best_both_r    = -float("inf")
        self.best_both_info = None
        self.n_both = 0
        self.n_sat  = 0

        _ep_path = os.path.join(run_dir, f"run{run_id}_episodes.csv")
        self._ep_f = open(_ep_path, 'w', newline='')
        self._ep_w = csv.writer(self._ep_f)
        self._ep_w.writerow(['ep', 'reward', 'w', 'l', 'dw', 'gap',
                             'sat', 'close', 'tight', 'imp', 'both'])

        _imp_path = os.path.join(run_dir, f"run{run_id}_improvements.csv")
        self._imp_f = open(_imp_path, 'w', newline='')
        self._imp_w = csv.writer(self._imp_f)
        self._imp_w.writerow(
            ['ep', 'tag', 'reward', 'w', 'l', 'gap']
            + [f'p{i}' for i in range(n_params)])
    # Add one episode record and update best-design trackers.
    def add(self, ep, r, info, params):
        self._ep_w.writerow([
            ep, f"{r:.6f}",
            f"{info['w']:.4f}", f"{info['l']:.4f}",
            f"{info['dw']:.6f}", f"{info['gap']:.6f}",
            int(info['sat']), int(info['close']), int(info['tight']),
            int(info['imp']), int(info['both']),
        ])
        if info['sat']:  self.n_sat  += 1
        if info['both']: self.n_both += 1

        _pl     = params.tolist() if hasattr(params, 'tolist') else list(params)
        _new_b  = r > self.best_r
        _new_bb = info['both'] and r > self.best_both_r

        def _row(tag):
            return ([ep, tag, f"{r:.6f}",
                     f"{info['w']:.4f}", f"{info['l']:.4f}", f"{info['gap']:.6f}"]
                    + [f"{v:.6f}" for v in _pl])

        if _new_b:
            self.best_r    = r
            self.best_info = dict(ep=ep, r=r, p=_pl, **info)
            self._imp_w.writerow(_row('best'))
        if _new_bb:
            self.best_both_r    = r
            self.best_both_info = dict(ep=ep, r=r, p=_pl, **info)
            self._imp_w.writerow(_row('both'))

        _tag = ""
        if _new_b and _new_bb: _tag = "  best+both"
        elif _new_b:            _tag = "  best"
        elif _new_bb:           _tag = "  both-best"
        if _tag:
            print(f"{_tag}  ep={ep}  r={r:.4f}  w={info['w']:.2f}  "
                  f"l={info['l']:.2f}  gap={info['gap']:.4f}")
    # Flush open CSV streams periodically during long experiments.
    def flush(self):
        self._ep_f.flush()
        self._imp_f.flush()
    # Close all log files at the end of a run.
    def close(self):
        self._ep_f.close()
        self._imp_f.close()

# Single-step rollout: one policy action proposes one complete design.
def _episode_single(env, sac, log, ep):
    s              = env.reset()
    a              = sac.select_action(s)
    s2, r, _, info = env.step(a)
    sac.rb.push(s, a, r, s2, 1.0)
    sac.update()
    log.add(ep, r, info, env.p)
    return r, info, a

# Multi-step rollout: several policy actions sequentially refine one design.
def _episode_multi(env, sac, log, ep, horizon):
    s       = env.reset()
    _tot    = 0.0
    _last_a = None
    _last_i = None
    for h in range(horizon):
        a              = sac.select_action(s)
        _done          = float(h == horizon - 1)
        s2, r, _, info = env.step(a)
        sac.rb.push(s, a, r, s2, _done)
        sac.update()
        _tot   += r
        _last_a = a
        _last_i = info
        s       = s2
    log.add(ep, _tot, _last_i, env.p)
    return _tot, _last_i, _last_a

# Train one independent run under a fixed seed.
# This function assembles the feature extractor, surrogate predictor, graph
# builder, environment, SAC agent, and logger.
def train_b(run_dir, run_id, seed, cfg):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _ext  = GraphFeatureExtractor()
    _pred = LoadPredictor(model_dir=cfg.model_dir, extractor=_ext)
    _bldr = GraphBuilder(s=cfg.gb_s, k=cfg.gb_k, t=cfg.gb_t, sc=cfg.gb_sc)
    _base = _resolve_base(cfg, seed)
    _env  = WeldEnvB(_bldr, _pred, _base, cfg)

    print(f"[R{run_id}] base = [{', '.join(f'{v:+.2f}' for v in _base)}]")
    print(f"[R{run_id}] w0={_env.w0:.2f}  l0={_env.l0:.2f}  "
          f"obs={_env.obs_dim}  act={_env.act_dim}")
    print(f"[R{run_id}] reward={cfg.reward_mode}"
          + (f"/{cfg.constrained_type}" if cfg.reward_mode == "constrained" else "")
          + f"  state={cfg.state_mode}  sac={cfg.sac_mode}"
          + f"  constr={cfg.constraint_mode}")
    print("=" * 90)

    _sac = SAC(
        obs_dim=_env.obs_dim,      act_dim=_env.act_dim,
        lr=cfg.lr,                 gamma=cfg.gamma,
        tau=cfg.tau,               alpha_init=cfg.alpha_init,
        auto_alpha=cfg.auto_alpha, batch_size=cfg.batch_size,
        warmup=cfg.warmup,         utd=cfg.utd,
        buf_cap=cfg.buf_cap,       hidden_dim=cfg.hidden_dim,
    )
    _log = LoggerB(run_dir, run_id, _env.w0, _env.l0, n_params=_env.dim)

    _itr = range(1, cfg.ep_max + 1)
    if not cfg.debug:
        _itr = tqdm(_itr, desc=f"Run {run_id}", leave=True)

    for ep in _itr:
        if cfg.sac_mode == "single_step":
            r, info, a = _episode_single(_env, _sac, _log, ep)
        else:
            r, info, a = _episode_multi(_env, _sac, _log, ep, cfg.horizon)

        if cfg.debug:
            _a_s  = ','.join(f'{v:+.3f}' for v in a)
            _p_s  = ','.join(f'{v:+.3f}' for v in _env.p)
            _tier = ("T" if info["tight"] else
                     "C" if info["close"] else
                     "S" if info["sat"]   else "-")
            print(f"  ep={ep:04d}  r={r:+8.4f}  "
                  f"w={info['w']:7.2f}(dw{info['dw']:+.4f})  "
                  f"l={info['l']:7.2f}(gap{info['gap']:.4f})  "
                  f"[{_tier}] imp={'Y' if info['imp'] else 'N'} "
                  f"both={'Y' if info['both'] else 'N'}  "
                  f"alpha={_sac.alpha:.4f}")
            if ep <= 20 or ep % 100 == 0:
                print(f"         act=[{_a_s}]")
                print(f"         par=[{_p_s}]")
        elif hasattr(_itr, 'set_postfix') and ep % 50 == 0:
            _itr.set_postfix(
                r=f"{r:+.3f}",
                best=f"{_log.best_r:.3f}",
                sat=f"{_log.n_sat/ep:.0%}",
                both=f"{_log.n_both/ep:.0%}")

        if ep % 500 == 0:
            _log.flush()

    _log.close()
    return _log

# Main driver for repeated SAC-B experiments.
# It creates a timestamped output directory, runs all seeds, summarizes results,
# and writes configuration and summary files for reproducibility.
def main_b():
    RunConfig.validate()
    cfg  = RunConfig
    _sep = "=" * 90

    _ts      = time.strftime("%Y%m%d_%H%M%S")
    _run_dir = _ensure_dir(os.path.join(cfg.output_dir, f"sacB_{_ts}"))

    print(_sep)
    print(f"  SAC-B  {cfg.n_runs} runs x {cfg.ep_max} eps"
          f"  reward={cfg.reward_mode}"
          + (f"/{cfg.constrained_type}" if cfg.reward_mode == "constrained" else "")
          + f"  state={cfg.state_mode}  sac={cfg.sac_mode}"
          + f"  constr={cfg.constraint_mode}"
          + ("  [DEBUG]" if cfg.debug else ""))
    print(f"  output -> {_run_dir}")
    print(_sep)

    _all = []
    for i in range(cfg.n_runs):
        _seed = i * 137 + cfg.seed_base
        print(f"\n{'─'*90}")
        print(f"  Run {i+1}/{cfg.n_runs}   seed={_seed}")
        print(f"{'─'*90}")

        _lg = train_b(_run_dir, run_id=i + 1, seed=_seed, cfg=cfg)
        _all.append(_lg)

        _b  = _lg.best_info
        _bb = _lg.best_both_info
        print(f"\n  Run {i+1} done:")
        print(f"    best : r={_b['r']:+.4f}  w={_b['w']:.2f}  "
              f"l={_b['l']:.2f}  ep={_b['ep']}")
        print("    both : " + (
            f"r={_bb['r']:+.4f}  w={_bb['w']:.2f}  "
            f"l={_bb['l']:.2f}  ep={_bb['ep']}" if _bb else "N/A"))
        print(f"    rates: sat={_lg.n_sat/cfg.ep_max:.1%}  "
              f"both={_lg.n_both/cfg.ep_max:.1%}")

    print(f"\n\n{_sep}\n  Summary\n{_sep}")
    print(f" Run | {'best_r':>9} {'w':>8} {'l':>8} | "
          f"{'both_r':>9} {'w':>8} {'l':>8} | "
          f"{'sat':>5} {'both':>5}")
    print("─" * 90)

    for idx, _lg in enumerate(_all):
        _b  = _lg.best_info
        _bb = _lg.best_both_info
        _br = f"{_bb['r']:9.4f}" if _bb else "      N/A"
        _bw = f"{_bb['w']:8.2f}" if _bb else "     N/A"
        _bl = f"{_bb['l']:8.2f}" if _bb else "     N/A"
        print(f"  {idx+1:2d} | "
              f"{_b['r']:9.4f} {_b['w']:8.2f} {_b['l']:8.2f} | "
              f"{_br} {_bw} {_bl} | "
              f"{_lg.n_sat/cfg.ep_max:5.1%} {_lg.n_both/cfg.ep_max:5.1%}")

    _brs = [_lg.best_r for _lg in _all]
    print(f"\n  best_r: {np.mean(_brs):.4f} +/- {np.std(_brs):.4f}  "
          f"({np.min(_brs):.4f} ~ {np.max(_brs):.4f})")

    _valid = [(idx, _lg) for idx, _lg in enumerate(_all)
              if _lg.best_both_info is not None]

    if _valid:
        _ci, _cl = max(_valid, key=lambda x: x[1].best_both_r)
        _ch      = _cl.best_both_info
        _p       = np.array(_ch['p'])
        _bref    = np.array(_resolve_base(cfg, _ci * 137 + cfg.seed_base))
        _delta   = _p - _bref
        print(f"\n  Champion (Run {_ci+1})")
        print(f"    r = {_ch['r']:+.4f}")
        print(f"    w = {_ch['w']:.2f}")
        print(f"    l = {_ch['l']:.2f}")
        print(f"    d = [{', '.join(f'{d:+.4f}' for d in _delta)}]")
        print(f"    p = [{', '.join(f'{v:+.4f}' for v in _p)}]")
    else:
        _bi = int(np.argmax(_brs))
        _b  = _all[_bi].best_info
        print(f"\n  No both-solution.  best Run {_bi+1}: "
              f"w={_b['w']:.2f}  l={_b['l']:.2f}")
    # Persist configuration and text summaries so that generated CSV logs remain
    # interpretable without relying on console output.
    with open(os.path.join(_run_dir, "config.json"), 'w', encoding='utf-8') as f:
        json.dump(cfg.as_dict(), f, indent=2, default=str)

    with open(os.path.join(_run_dir, "summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"SAC-B  {cfg.n_runs} runs x {cfg.ep_max} eps\n")
        f.write(f"reward={cfg.reward_mode}"
                + (f"/{cfg.constrained_type}" if cfg.reward_mode == "constrained" else "")
                + f"  state={cfg.state_mode}  sac={cfg.sac_mode}"
                + f"  constr={cfg.constraint_mode}\n\n")
        for idx, _lg in enumerate(_all):
            _b  = _lg.best_info
            _bb = _lg.best_both_info
            f.write(f"Run {idx+1}:\n")
            f.write(f"  best:  r={_b['r']:+.4f}  w={_b['w']:.2f}  "
                    f"l={_b['l']:.2f}  ep={_b['ep']}\n")
            if _bb:
                f.write(f"  both:  r={_bb['r']:+.4f}  w={_bb['w']:.2f}  "
                        f"l={_bb['l']:.2f}  ep={_bb['ep']}\n")
                f.write(f"  params: {_bb['p']}\n")
            else:
                f.write("  both:  N/A\n")
            f.write(f"  sat={_lg.n_sat/cfg.ep_max:.1%}  "
                    f"both={_lg.n_both/cfg.ep_max:.1%}\n\n")
        if _valid:
            f.write(f"\nChampion: Run {_ci+1}\n")
            f.write(f"  r={_ch['r']:+.4f}  w={_ch['w']:.2f}  l={_ch['l']:.2f}\n")
            f.write(f"  params={_ch['p']}\n")

    print(f"\n  saved -> {_run_dir}/")
    print(f"    config.json")
    print(f"    run*_episodes.csv")
    print(f"    run*_improvements.csv")
    print(f"    summary.txt")
    print(_sep)

    return _all


if __name__ == "__main__":
    main_b()