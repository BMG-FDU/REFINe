import json, copy, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm
from tqdm import tqdm

warnings.filterwarnings("ignore")


class CFG:
    CSV_PATH          = None
    GRAPH_ROOT        = None
    SAVE_DIR          = None

    CLEAN_PCT         = None
    CLEAN_MIN_SAMPLES = None
    N_FOLDS           = None
    SEED              = None

    SAVE_MODE         = None
    SPLIT_MODE        = None
    HOLDOUT_RATIO     = None

    OPTIMIZER         = None
    PATIENCE          = None
    LR                = None
    BATCH_SIZE        = None

    GIN_HIDDEN        = None
    GIN_LAYERS        = None
    GIN_DROPOUT       = None
    GIN_MAX_EPOCHS    = None
    GIN_GRAD_CLIP     = None
    GIN_WD            = None

    ET_N              = None
    GBR_N             = None
    GBR_LR            = None
    GBR_DEPTH         = None
    GBR_SUB           = None

    DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s=CFG.SEED):
    np.random.seed(s)
    torch.manual_seed(s if s is not None else 0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s if s is not None else 0)


class GINRegressor(nn.Module):
    def __init__(self, node_dim, global_dim):
        hidden   = CFG.GIN_HIDDEN  if CFG.GIN_HIDDEN  is not None else 16
        n_layers = CFG.GIN_LAYERS  if CFG.GIN_LAYERS  is not None else 1
        dropout  = CFG.GIN_DROPOUT if CFG.GIN_DROPOUT is not None else 0.5
        super().__init__()
        self.dropout = dropout
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            d_in = node_dim if i == 0 else hidden
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))))
            self.bns.append(BatchNorm(hidden))
        self.pool = global_mean_pool
        self.gf_bn = nn.BatchNorm1d(global_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden + global_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, max(1, hidden // 2)), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(max(1, hidden // 2), 1))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.dropout(F.relu(bn(conv(x, ei))), p=self.dropout, training=self.training)
        g = self.pool(x, batch)
        gf = data.global_feat.squeeze(1) if data.global_feat.dim() == 3 else data.global_feat
        return self.head(torch.cat([g, self.gf_bn(gf)], dim=-1)).squeeze(-1)


class Pipeline:

    def __init__(self):
        self.X = self.y = self.keys = self.feature_cols = None
        self.pyg_data = None
        self.clean_idx = None
        self.deploy_w_et = 0.5

    def _load_graph(self, fp):
        raw = json.loads(Path(fp).read_text())
        G0 = nx.Graph()
        for n in raw["nodes"]:
            G0.add_node(n["id"], pos=tuple(n["pos"]))
        for e in raw["links"]:
            G0.add_edge(e["source"], e["target"])
        pos0 = nx.get_node_attributes(G0, "pos")
        c2id = defaultdict(list)
        for nid, p in pos0.items():
            c2id[p].append(nid)
        G = nx.Graph()
        coords = list(c2id.keys())
        for i, c in enumerate(coords):
            G.add_node(i, pos=c)
        lk = {old: coords.index(pos0[old]) for old in G0.nodes}
        for u, v in G0.edges:
            a, b = lk[u], lk[v]
            if a != b:
                G.add_edge(a, b)
        return G

    def _to_pyg(self, G, gvec, target):
        nodes = sorted(G.nodes())
        nmap = {n: i for i, n in enumerate(nodes)}
        pos = nx.get_node_attributes(G, "pos")
        clust = nx.clustering(G)
        nf = []
        for n in nodes:
            p = pos.get(n, [0., 0.])
            d = G.degree(n)
            nbs = list(G.neighbors(n))
            ael = 0.
            if nbs and n in pos:
                pn = np.array(pos[n], dtype=float)
                ael = np.mean([np.linalg.norm(
                    pn - np.array(pos.get(nb, [0., 0.]), dtype=float))
                    for nb in nbs])
            nf.append([p[0], p[1], float(d),
                       float(d == 1), float(d > 2), clust.get(n, 0.), ael])
        x = torch.tensor(nf, dtype=torch.float) if nf else torch.zeros(1, 7)
        el = []
        for u, v in G.edges():
            el += [[nmap[u], nmap[v]], [nmap[v], nmap[u]]]
        ei = (torch.tensor(el, dtype=torch.long).t().contiguous()
              if el else torch.zeros(2, 0, dtype=torch.long))
        return Data(x=x, edge_index=ei,
                    global_feat=torch.tensor(gvec, dtype=torch.float).unsqueeze(0),
                    y=torch.tensor([target], dtype=torch.float))

    def load_data(self):
        df = pd.read_csv(CFG.CSV_PATH).dropna(subset=["max_load"])
        self.feature_cols = [c for c in df.columns if c not in ("key", "max_load")]
        self.X = (df[self.feature_cols].fillna(0)
                  .replace([np.inf, -np.inf], 0).values.astype(np.float32))
        self.y = df["max_load"].values.astype(np.float32)
        self.keys = df["key"].values
        print(f"  samples={len(self.y)}  features={len(self.feature_cols)}D  "
              f"y=[{self.y.min():.1f}, {self.y.max():.1f}]")

        pyg, miss = [], 0
        for i in tqdm(range(len(self.keys)), desc="  building PyG graphs"):
            fp = CFG.GRAPH_ROOT / f"{self.keys[i]}.json"
            if fp.exists():
                pyg.append(self._to_pyg(self._load_graph(fp), self.X[i], self.y[i]))
            else:
                miss += 1
                pyg.append(Data(
                    x=torch.zeros(1, 7),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    global_feat=torch.tensor(self.X[i], dtype=torch.float).unsqueeze(0),
                    y=torch.tensor([self.y[i]], dtype=torch.float)))
        self.pyg_data = pyg
        if miss:
            print(f"  {miss} graph files missing, replaced with empty graphs")

    def clean_outliers(self):
        if CFG.CLEAN_MIN_SAMPLES is None:
            raise ValueError("Invalid config")
        if len(self.y) < CFG.CLEAN_MIN_SAMPLES:
            self.clean_idx = np.arange(len(self.y))
            return

        kf = KFold(CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
        oof = np.zeros(len(self.y))
        for tri, vai in kf.split(self.X):
            oof[vai] = ExtraTreesRegressor(
                n_estimators=500, random_state=CFG.SEED, n_jobs=-1
            ).fit(self.X[tri], self.y[tri]).predict(self.X[vai])
        res = np.abs(self.y - oof)
        n_rm = int(len(self.y) * CFG.CLEAN_PCT / 100)
        self.clean_idx = np.argsort(res)[:-n_rm] if n_rm > 0 else np.arange(len(self.y))
        print(f"  OOF R2={r2_score(self.y, oof):.4f}  "
              f"MAE={res.mean():.2f}  P95={np.quantile(res, .95):.2f}  MAX={res.max():.2f}")
        print(f"  removed {n_rm} -> remaining {len(self.clean_idx)}")

    def _build_optimizer(self, model):
        lr = CFG.LR    if CFG.LR    is not None else 1e-3
        wd = CFG.GIN_WD if CFG.GIN_WD is not None else 0.0
        if CFG.OPTIMIZER == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.Adam(model.parameters(), lr=lr)

    def _resolve_tree_params(self):
        return dict(
            et_n    = CFG.ET_N      if CFG.ET_N      is not None else 10,
            gbr_n   = CFG.GBR_N    if CFG.GBR_N    is not None else 10,
            gbr_lr  = CFG.GBR_LR   if CFG.GBR_LR   is not None else 0.1,
            gbr_d   = CFG.GBR_DEPTH if CFG.GBR_DEPTH is not None else 3,
            gbr_sub = CFG.GBR_SUB  if CFG.GBR_SUB  is not None else 1.0,
        )

    def _train_gin(self, nd, gd, dataset, tri, vai, eval_idx=None):
        if eval_idx is None:
            eval_idx = vai
        batch_size = CFG.BATCH_SIZE     if CFG.BATCH_SIZE     is not None else 16
        patience   = CFG.PATIENCE       if CFG.PATIENCE       is not None else 99999
        max_ep     = CFG.GIN_MAX_EPOCHS if CFG.GIN_MAX_EPOCHS is not None else 50
        grad_clip  = CFG.GIN_GRAD_CLIP  if CFG.GIN_GRAD_CLIP  is not None else 1.0

        tr_ld = DataLoader([dataset[i] for i in tri], batch_size, shuffle=True)
        va_ld = DataLoader([dataset[i] for i in vai], batch_size)
        ev_ld = DataLoader([dataset[i] for i in eval_idx], batch_size)
        model = GINRegressor(nd, gd).to(CFG.DEVICE)
        opt = self._build_optimizer(model)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", factor=0.5, patience=10, min_lr=1e-6)
        best_r2, best_st, wait = -1e9, None, 0
        for ep in range(1, max_ep + 1):
            model.train()
            ep_loss = 0.
            for b in tr_ld:
                b = b.to(CFG.DEVICE)
                opt.zero_grad()
                loss = F.mse_loss(model(b), b.y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_loss += loss.item() * b.num_graphs
            sch.step(ep_loss / len(tri))
            if ep % 5 == 0:
                model.eval()
                ps, yv = [], []
                with torch.no_grad():
                    for b in va_ld:
                        b = b.to(CFG.DEVICE)
                        ps.append(model(b).cpu().numpy())
                        yv.append(b.y.cpu().numpy())
                r2 = r2_score(np.concatenate(yv), np.concatenate(ps))
                if r2 > best_r2:
                    best_r2, best_st, wait = r2, copy.deepcopy(model.state_dict()), 0
                else:
                    wait += 5
                if wait >= patience:
                    break
        if best_st:
            model.load_state_dict(best_st)
        model.eval()
        ps = []
        with torch.no_grad():
            for b in ev_ld:
                b = b.to(CFG.DEVICE)
                ps.append(model(b).cpu().numpy())
        return np.concatenate(ps)

    def _train_gin_full(self, nd, gd, dataset):
        seed       = CFG.SEED         if CFG.SEED         is not None else 0
        batch_size = CFG.BATCH_SIZE   if CFG.BATCH_SIZE   is not None else 16
        patience   = CFG.PATIENCE     if CFG.PATIENCE     is not None else 99999
        max_ep     = CFG.GIN_MAX_EPOCHS if CFG.GIN_MAX_EPOCHS is not None else 50
        grad_clip  = CFG.GIN_GRAD_CLIP  if CFG.GIN_GRAD_CLIP  is not None else 1.0

        n = len(dataset)
        n_val = max(1, int(n * 0.1))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n).tolist()
        vai, tri = perm[:n_val], perm[n_val:]

        tr_ld = DataLoader([dataset[i] for i in tri], batch_size, shuffle=True)
        va_ld = DataLoader([dataset[i] for i in vai], batch_size)

        model = GINRegressor(nd, gd).to(CFG.DEVICE)
        opt = self._build_optimizer(model)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", factor=0.5, patience=10, min_lr=1e-6)

        best_r2, best_ep, wait = -1e9, 1, 0

        for ep in range(1, max_ep + 1):
            model.train()
            ep_loss = 0.0
            for b in tr_ld:
                b = b.to(CFG.DEVICE)
                opt.zero_grad()
                loss = F.mse_loss(model(b), b.y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                ep_loss += loss.item() * b.num_graphs
            sch.step(ep_loss / len(tri))
            if ep % 5 == 0:
                model.eval()
                ps, yv = [], []
                with torch.no_grad():
                    for b in va_ld:
                        b = b.to(CFG.DEVICE)
                        ps.append(model(b).cpu().numpy())
                        yv.append(b.y.cpu().numpy())
                r2 = r2_score(np.concatenate(yv), np.concatenate(ps))
                if r2 > best_r2:
                    best_r2, best_ep, wait = r2, ep, 0
                else:
                    wait += 5
                if wait >= patience:
                    break

        all_ld = DataLoader(dataset, batch_size, shuffle=True)
        model_f = GINRegressor(nd, gd).to(CFG.DEVICE)
        opt2 = self._build_optimizer(model_f)
        sch2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt2, "min", factor=0.5, patience=10, min_lr=1e-6)

        pbar = tqdm(range(1, best_ep + 1), desc="  GIN full", leave=False)
        for ep in pbar:
            model_f.train()
            ep_loss = 0.0
            for b in all_ld:
                b = b.to(CFG.DEVICE)
                opt2.zero_grad()
                loss = F.mse_loss(model_f(b), b.y)
                loss.backward()
                nn.utils.clip_grad_norm_(model_f.parameters(), grad_clip)
                opt2.step()
                ep_loss += loss.item() * b.num_graphs
            sch2.step(ep_loss / n)
            pbar.set_postfix(loss=f"{ep_loss / n:.4f}")

        model_f.eval()
        return model_f

    @staticmethod
    def _search_w3(p1, p2, p3, yt):
        best, bw = -1e9, (1/3, 1/3, 1/3)
        for w1 in np.arange(0.05, 0.85, 0.05):
            for w2 in np.arange(0.05, 0.90 - w1, 0.05):
                w3 = 1.0 - w1 - w2
                if w3 < 0.05:
                    continue
                s = r2_score(yt, w1 * p1 + w2 * p2 + w3 * p3)
                if s > best:
                    best, bw = s, (round(w1, 2), round(w2, 2), round(w3, 2))
        return best, bw

    @staticmethod
    def _search_w2(p1, p2, yt):
        best, bw = -1e9, 0.5
        for w in np.arange(0.05, 0.96, 0.05):
            s = r2_score(yt, w * p1 + (1 - w) * p2)
            if s > best:
                best, bw = s, round(w, 2)
        return best, bw

    def _evaluate_kfold(self, cX, cy, cpyg):
        nd, gd = cpyg[0].x.shape[1], cX.shape[1]
        kf = KFold(CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
        s_et, s_gb, s_gin, s_3m, s_2m, w2s = [], [], [], [], [], []
        tp = self._resolve_tree_params()

        pbar = tqdm(list(kf.split(cX)), desc="  CV")
        for fold, (tri, vai) in enumerate(pbar):
            pbar.set_description(f"  CV Fold {fold + 1}/{CFG.N_FOLDS}")
            set_seed((CFG.SEED if CFG.SEED is not None else 0) + fold)
            yv = cy[vai]

            p_et = ExtraTreesRegressor(
                n_estimators=tp["et_n"], random_state=CFG.SEED, n_jobs=-1
            ).fit(cX[tri], cy[tri]).predict(cX[vai])

            p_gb = GradientBoostingRegressor(
                n_estimators=tp["gbr_n"], learning_rate=tp["gbr_lr"],
                max_depth=tp["gbr_d"], subsample=tp["gbr_sub"], random_state=CFG.SEED
            ).fit(cX[tri], cy[tri]).predict(cX[vai])

            p_gin = self._train_gin(nd, gd, cpyg, tri, vai)

            re, rg, ri = r2_score(yv, p_et), r2_score(yv, p_gb), r2_score(yv, p_gin)
            s_et.append(re); s_gb.append(rg); s_gin.append(ri)

            r3, w3 = self._search_w3(p_et, p_gb, p_gin, yv)
            s_3m.append(r3)

            r2_s, w2 = self._search_w2(p_et, p_gb, yv)
            s_2m.append(r2_s); w2s.append(w2)

            tqdm.write(
                f"  Fold {fold + 1}: ET={re:.4f}  GBR={rg:.4f}  GIN={ri:.4f}  "
                f"3M={r3:.4f} w={list(w3)}  2M={r2_s:.4f} w_et={w2}")

        self.deploy_w_et = float(np.mean(w2s))
        print(f"\n  ET  = {np.mean(s_et):.4f} +/- {np.std(s_et):.4f}")
        print(f"  GBR = {np.mean(s_gb):.4f} +/- {np.std(s_gb):.4f}")
        print(f"  GIN = {np.mean(s_gin):.4f} +/- {np.std(s_gin):.4f}")
        print(f"\n  best R2 (ET+GBR+GIN): {np.mean(s_3m):.4f} +/- {np.std(s_3m):.4f}")
        print(f"  deploy R2 (ET+GBR):   {np.mean(s_2m):.4f} +/- {np.std(s_2m):.4f}")
        print(f"  deploy weights: ET={self.deploy_w_et:.2f}  GBR={1 - self.deploy_w_et:.2f}")

    def _evaluate_holdout(self, cX, cy, cpyg):
        if CFG.HOLDOUT_RATIO is None:
            raise ValueError("Invalid config")
        nd, gd = cpyg[0].x.shape[1], cX.shape[1]
        n = len(cy)
        r = CFG.HOLDOUT_RATIO
        rng = np.random.default_rng(CFG.SEED if CFG.SEED is not None else 0)
        idx = rng.permutation(n)
        n_tr = int(n * r[0])
        n_va = int(n * r[1])
        tri = idx[:n_tr]
        vai = idx[n_tr:n_tr + n_va]
        tei = idx[n_tr + n_va:]
        tp = self._resolve_tree_params()

        set_seed(CFG.SEED if CFG.SEED is not None else 0)

        p_et = ExtraTreesRegressor(
            n_estimators=tp["et_n"], random_state=CFG.SEED, n_jobs=-1
        ).fit(cX[tri], cy[tri]).predict(cX[tei])

        p_gb = GradientBoostingRegressor(
            n_estimators=tp["gbr_n"], learning_rate=tp["gbr_lr"],
            max_depth=tp["gbr_d"], subsample=tp["gbr_sub"], random_state=CFG.SEED
        ).fit(cX[tri], cy[tri]).predict(cX[tei])

        p_gin = self._train_gin(nd, gd, cpyg, tri, vai, eval_idx=tei)

        yt = cy[tei]
        re = r2_score(yt, p_et)
        rg = r2_score(yt, p_gb)
        ri = r2_score(yt, p_gin)

        r3, w3 = self._search_w3(p_et, p_gb, p_gin, yt)
        r2_s, w2 = self._search_w2(p_et, p_gb, yt)
        self.deploy_w_et = w2

        print(f"  ET={re:.4f}  GBR={rg:.4f}  GIN={ri:.4f}")
        print(f"  best R2 (ET+GBR+GIN): {r3:.4f} w={list(w3)}")
        print(f"  deploy R2 (ET+GBR):   {r2_s:.4f}  w_et={w2}")
        print(f"  deploy weights: ET={self.deploy_w_et:.2f}  GBR={1 - self.deploy_w_et:.2f}")

    def evaluate(self):
        idx = self.clean_idx
        cX, cy = self.X[idx], self.y[idx]
        cpyg = [self.pyg_data[i] for i in idx]
        if CFG.SPLIT_MODE == "kfold":
            self._evaluate_kfold(cX, cy, cpyg)
        elif CFG.SPLIT_MODE == "holdout":
            self._evaluate_holdout(cX, cy, cpyg)
        else:
            raise ValueError("Invalid SPLIT_MODE")

    def train_and_save(self):
        if CFG.SAVE_MODE not in ("trees_only", "full"):
            raise ValueError("Invalid SAVE_MODE")

        idx = self.clean_idx
        cX, cy = self.X[idx], self.y[idx]
        d = CFG.SAVE_DIR
        d.mkdir(parents=True, exist_ok=True)
        tp = self._resolve_tree_params()

        et = ExtraTreesRegressor(
            n_estimators=tp["et_n"], random_state=CFG.SEED, n_jobs=-1
        ).fit(cX, cy)
        gbr = GradientBoostingRegressor(
            n_estimators=tp["gbr_n"], learning_rate=tp["gbr_lr"],
            max_depth=tp["gbr_d"], subsample=tp["gbr_sub"], random_state=CFG.SEED
        ).fit(cX, cy)

        joblib.dump(et,  d / "et_model.joblib")
        joblib.dump(gbr, d / "gbr_model.joblib")

        cfg = dict(
            mode=CFG.SAVE_MODE,
            feature_cols=self.feature_cols,
            w_et=round(self.deploy_w_et, 4),
            w_gbr=round(1 - self.deploy_w_et, 4),
            clean_pct=CFG.CLEAN_PCT,
            n_features=len(self.feature_cols),
            n_train=int(len(idx)),
        )

        if CFG.SAVE_MODE == "full":
            cpyg = [self.pyg_data[i] for i in idx]
            nd = cpyg[0].x.shape[1]
            gd = cX.shape[1]
            gin_model = self._train_gin_full(nd, gd, cpyg)
            torch.save(gin_model.state_dict(), d / "gin_model.pt")
            cfg["gin_nd"]      = int(nd)
            cfg["gin_gd"]      = int(gd)
            cfg["gin_hidden"]  = int(CFG.GIN_HIDDEN  if CFG.GIN_HIDDEN  is not None else 16)
            cfg["gin_layers"]  = int(CFG.GIN_LAYERS  if CFG.GIN_LAYERS  is not None else 1)
            cfg["gin_dropout"] = float(CFG.GIN_DROPOUT if CFG.GIN_DROPOUT is not None else 0.5)

        (d / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

        p = self.deploy_w_et * et.predict(cX) + (1 - self.deploy_w_et) * gbr.predict(cX)
        print(f"  train R2={r2_score(cy, p):.4f}  samples={len(idx)}")
        print(f"  saved to: {d}/")
        print(f"    et_model.joblib")
        print(f"    gbr_model.joblib")
        if CFG.SAVE_MODE == "full":
            print(f"    gin_model.pt")
        print(f"    config.json")

    def run(self):
        set_seed()
        print(f"  Device: {CFG.DEVICE}")
        self.load_data()
        self.clean_outliers()
        self.evaluate()
        self.train_and_save()


Pipeline().run()