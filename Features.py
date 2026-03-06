"""
graph_feature_extractor.py
Unified 94-dimensional feature extraction for weld graph structures.
  - 34 structural / topological features
  - 18 pore features  (original 10 + new 8)
  - 42 contact (overlap) features
"""

from pathlib import Path
from collections import defaultdict
import json, math, datetime, warnings
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy.stats import entropy, linregress, skew as sp_skew, kurtosis as sp_kurtosis
from scipy.sparse.linalg import eigsh
import cv2

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GraphFeatureExtractor:

    FEATURE_COLS = [
        # Structural (34)
        "n_node", "n_edge", "total_length", "mean_edge_len", "len_cv",
        "deg2_count", "deg4_count", "degree_entropy",
        "orient_entropy", "anisotropy",
        "radius_gyration", "moment_total",
        "clustering_coef", "fiedler_value", "lambda_max", "spectral_gap_ratio", "aspl_giant",
        "boundary_frac", "fractal_dim_box",
        "rigidity_index", "redundancy_ratio", "max_k_core", "kcore_frac",
        "triangle_count", "triangle_ratio", "quad_count", "quad_ratio",
        "avg_shortest_dy", "straightness",
        "mesh_median_area", "mesh_cv_area", "mesh_max_area_ratio",
        "edge_betweenness_max", "edge_betweenness_gini",
        # Pore (18 = 10 original + 8 new)
        "largest_pore_ratio", "top_area_sum_ratio",
        "top_convexity_min", "top_convexity_mean", "top_circularity_min",
        "big_pore_count", "total_pore_count",
        "total_pore_ratio", "center_pore_ratio", "edge_pore_ratio",
        "pore_area_cv", "pore_area_skew", "pore_area_kurtosis",
        "pore_area_max_over_mean", "pore_large_area_frac", "pore_count_large_frac",
        "pore_density", "pore_spatial_cv",
        # Contact (42)
        "contact_thick", "contact_canvas_size", "contact_nodes", "contact_edges",
        "contact_edge_pixel_union_count", "contact_edge_pixel_sum",
        "contact_raw_overlap_pixel_count", "contact_overlap_pixel_count",
        "contact_overlap_pair_count", "contact_overlap_pairs_per_edge",
        "contact_edges_with_contact_count", "contact_edges_with_contact_ratio",
        "contact_overlap_pixel_ratio_union", "contact_raw_overlap_pixel_ratio_union",
        "contact_overlap_pixel_ratio_canvas",
        "contact_overlap_length_px_approx", "contact_overlap_length_ratio_centerline",
        "contact_centerline_length_px",
        "contact_overlap_pair_size_sum", "contact_overlap_pair_size_mean",
        "contact_overlap_pair_size_median", "contact_overlap_pair_size_max",
        "contact_overlap_pair_size_std", "contact_overlap_pair_size_q75",
        "contact_overlap_pair_size_q90", "contact_overlap_pair_size_q95",
        "contact_overlap_cc_count", "contact_overlap_cc_size_sum",
        "contact_overlap_cc_size_mean", "contact_overlap_cc_size_median",
        "contact_overlap_cc_size_max", "contact_overlap_cc_size_std",
        "contact_overlap_cc_size_q75", "contact_overlap_cc_size_q90",
        "contact_overlap_cc_size_q95",
        "contact_edge_contact_degree_mean", "contact_edge_contact_degree_median",
        "contact_edge_contact_degree_max", "contact_edge_contact_degree_std",
        "contact_edge_contact_degree_q75", "contact_edge_contact_degree_q90",
        "contact_edge_contact_degree_q95",
    ]

    def __init__(self, root, canvas_size=1024, thick=9,
                 edge_margin=0.12, top_k=3, area_thresh=0.005, connectivity=8):
        self.root = Path(root)
        self.canvas_size = canvas_size
        self.thick = thick
        self.edge_margin = edge_margin
        self.top_k = top_k
        self.area_thresh = area_thresh
        self.connectivity = connectivity

    # ==================== Utilities ====================

    @staticmethod
    def _gini(x):
        if x.size == 0 or x.sum() == 0:
            return 0.0
        x = np.sort(x)
        n = x.size
        c = np.cumsum(x, dtype=float)
        return (n + 1 - 2 * (c / c[-1]).sum()) / n

    @staticmethod
    def _safe_stats(arr):
        if len(arr) == 0:
            return dict(count=0, sum=0.0, mean=0.0, median=0.0, max=0.0,
                        std=0.0, q75=0.0, q90=0.0, q95=0.0)
        a = np.asarray(arr, dtype=float)
        return dict(
            count=int(a.size), sum=float(a.sum()),
            mean=float(a.mean()), median=float(np.median(a)),
            max=float(a.max()), std=float(a.std(ddof=0)),
            q75=float(np.quantile(a, 0.75)),
            q90=float(np.quantile(a, 0.90)),
            q95=float(np.quantile(a, 0.95)),
        )

    @staticmethod
    def _cc_sizes_from_pixels(pixel_set, connectivity=8):
        if not pixel_set:
            return []
        visited = set()
        sizes = []
        nbr = ([(1, 0), (-1, 0), (0, 1), (0, -1)] if connectivity == 4
               else [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)])
        for px in pixel_set:
            if px in visited:
                continue
            stack = [px]
            visited.add(px)
            sz = 0
            while stack:
                y, x = stack.pop()
                sz += 1
                for dy, dx in nbr:
                    nb = (y + dy, x + dx)
                    if nb in pixel_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            sizes.append(sz)
        return sizes

    @staticmethod
    def _get_edge_pixels(pt1, pt2, thick):
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        x2, y2 = int(round(pt2[0])), int(round(pt2[1]))
        m = thick + 2
        min_x, max_x = max(0, min(x1, x2) - m), max(x1, x2) + m
        min_y, max_y = max(0, min(y1, y2) - m), max(y1, y2) + m
        h, w = max_y - min_y + 1, max_x - min_x + 1
        if h <= 0 or w <= 0:
            return set()
        buf = np.zeros((h, w), dtype=np.uint8)
        cv2.line(buf, (x1 - min_x, y1 - min_y), (x2 - min_x, y2 - min_y),
                 255, thick, cv2.LINE_AA)
        ys, xs = np.where(buf > 0)
        return {(int(y + min_y), int(x + min_x)) for y, x in zip(ys, xs)}

    # ==================== Graph I/O ====================

    def _load_graph(self, fp):
        data = json.loads(Path(fp).read_text())
        G0 = nx.Graph()
        for n in data["nodes"]:
            G0.add_node(n["id"], pos=tuple(n["pos"]))
        for e in data["links"]:
            G0.add_edge(e["source"], e["target"])
        pos0 = nx.get_node_attributes(G0, "pos")
        coord2ids = defaultdict(list)
        for nid, p in pos0.items():
            coord2ids[p].append(nid)
        G = nx.Graph()
        coords = list(coord2ids.keys())
        for i, c in enumerate(coords):
            G.add_node(i, pos=c)
        lookup = {old: coords.index(pos0[old]) for old in G0.nodes}
        for u, v in G0.edges:
            a, b = lookup[u], lookup[v]
            if a != b:
                G.add_edge(a, b)
        return G

    def _render_image(self, G):
        pos = np.array([G.nodes[n]["pos"] for n in G.nodes])
        min_xy, max_xy = pos.min(0), pos.max(0)
        span = (max_xy - min_xy).max() or 1
        scale = (self.canvas_size - 10) / span
        pts = ((pos - min_xy) * scale + 5).astype(int)
        id2pt = {n: tuple(p) for n, p in zip(G.nodes, pts)}
        img = np.ones((self.canvas_size, self.canvas_size), np.uint8) * 255
        for u, v in G.edges:
            cv2.line(img, id2pt[u], id2pt[v], 0, self.thick, cv2.LINE_AA)
        return img, id2pt

    # ==================== Structural Features (34) ====================

    def _basic_size(self, G):
        pos = nx.get_node_attributes(G, "pos")
        lengths = np.array([math.dist(pos[u], pos[v]) for u, v in G.edges], dtype=float)
        nn, ne = G.number_of_nodes(), G.number_of_edges()
        return dict(
            n_node=nn, n_edge=ne,
            total_length=float(lengths.sum()),
            mean_edge_len=float(lengths.mean() if lengths.size else 0.0),
            len_cv=float(lengths.std() / lengths.mean()) if (lengths.size and lengths.mean()) else 0.0,
        )

    def _degree_stats(self, G):
        deg = np.array([d for _, d in G.degree()], dtype=int)
        p = np.bincount(deg) / deg.size
        return dict(
            deg2_count=int((deg == 2).sum()),
            deg4_count=int((deg == 4).sum()),
            degree_entropy=float(entropy(p[p > 0], base=2)),
        )

    def _orientation_stats(self, G):
        pos = nx.get_node_attributes(G, "pos")
        ang = np.array([math.atan2(pos[v][1] - pos[u][1], pos[v][0] - pos[u][0])
                        for u, v in G.edges], dtype=float)
        if ang.size == 0:
            return dict(orient_entropy=0.0, anisotropy=0.0)
        bins = np.histogram(ang, bins=18, range=(-math.pi, math.pi))[0]
        oe = float(entropy(bins[bins > 0], base=2))
        cs = np.column_stack([np.cos(ang), np.sin(ang)])
        Q = (cs.T @ cs) / ang.size
        eig = np.linalg.eigvalsh(Q)
        ani = float((eig[1] - eig[0]) / (eig[1] + eig[0] + 1e-12))
        return dict(orient_entropy=oe, anisotropy=ani)

    def _spatial_moments(self, G):
        pos = np.array([G.nodes[n]["pos"] for n in G.nodes])
        cen = pos.mean(0)
        deg = np.array([d for _, d in G.degree()])
        return dict(
            radius_gyration=float(np.sqrt(((pos - cen) ** 2).sum(1).mean())),
            moment_total=float(np.sum(np.linalg.norm(pos - cen, axis=1) * deg)),
        )

    def _path_connectivity(self, G):
        cc_coef = float(nx.average_clustering(G))
        try:
            fiedler = float(nx.algebraic_connectivity(G))
        except nx.NetworkXError:
            fiedler = 0.0
        GC = G.subgraph(max(nx.connected_components(G), key=len))
        aspl = float(nx.average_shortest_path_length(GC)) if GC.number_of_nodes() > 1 else 0.0
        L = nx.laplacian_matrix(G).astype(float)
        try:
            lmax = float(eigsh(L, k=1, which="LA", return_eigenvectors=False)[0])
        except Exception:
            lmax = 0.0
        sgr = float(fiedler / (lmax + 1e-12)) if lmax else 0.0
        return dict(clustering_coef=cc_coef, fiedler_value=fiedler,
                    lambda_max=lmax, spectral_gap_ratio=sgr, aspl_giant=aspl)

    def _boundary_fractal(self, G):
        pos = np.array([G.nodes[n]["pos"] for n in G.nodes])
        ne = G.number_of_edges()
        eps = 1e-6
        xmin, xmax = pos[:, 0].min(), pos[:, 0].max()
        ymin, ymax = pos[:, 1].min(), pos[:, 1].max()
        bn = {n for n, (x, y) in nx.get_node_attributes(G, "pos").items()
              if abs(x - xmin) < eps or abs(x - xmax) < eps
              or abs(y - ymin) < eps or abs(y - ymax) < eps}
        be = [(u, v) for u, v in G.edges if u in bn or v in bn]
        bf = len(be) / ne if ne else 0.0
        sizes, counts = [], []
        for k in range(1, 7):
            s = 1 / 2 ** k
            idx = np.floor(pos / s).astype(int)
            counts.append(len({tuple(i) for i in idx}))
            sizes.append(1 / s)
        fd = float(linregress(np.log(sizes), np.log(counts)).slope)
        return dict(boundary_frac=bf, fractal_dim_box=fd)

    def _redundancy_kcore(self, G):
        nn, ne = G.number_of_nodes(), G.number_of_edges()
        ri = float(ne - 2 * nn + 3)
        rr = float((ne - nn + 1) / ne) if ne else 0.0
        cn = nx.core_number(G)
        mk = max(cn.values())
        kf = float(sum(1 for v in cn.values() if v == mk) / nn)
        return dict(rigidity_index=ri, redundancy_ratio=rr, max_k_core=mk, kcore_frac=kf)

    def _cycle_features(self, G):
        ne = G.number_of_edges()
        tc = sum(nx.triangles(G).values()) // 3
        qc = len([c for c in nx.cycle_basis(G) if len(c) == 4])
        return dict(triangle_count=tc, triangle_ratio=float(tc / ne) if ne else 0.0,
                    quad_count=qc, quad_ratio=float(qc / ne) if ne else 0.0)

    def _vertical_shortestness(self, G):
        pos = nx.get_node_attributes(G, "pos")
        yv = np.array([p[1] for p in pos.values()])
        ymin, ymax = yv.min(), yv.max()
        eps = 1e-6
        top = [n for n, p in pos.items() if abs(p[1] - ymax) < eps]
        bot = [n for n, p in pos.items() if abs(p[1] - ymin) < eps]
        if not (top and bot):
            return dict(avg_shortest_dy=0.0, straightness=0.0)
        for u, v in G.edges:
            dy = abs(pos[u][1] - pos[v][1])
            G.edges[u, v]["w"] = dy if dy else 1e-6
        dists = []
        for s in top:
            d = nx.single_source_dijkstra_path_length(G, s, weight="w")
            dists.extend(d[t] for t in bot if t in d)
        avg = float(np.mean(dists)) if dists else 0.0
        return dict(avg_shortest_dy=avg, straightness=float(avg / (ymax - ymin + 1e-12)))

    def _mesh_holes(self, img):
        res = self.canvas_size
        work = img.copy()
        cv2.floodFill(work, None, (0, 0), 128)
        mask = work == 255
        if mask.sum() == 0:
            return dict(mesh_median_area=0.0, mesh_cv_area=0.0, mesh_max_area_ratio=0.0)
        _, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        areas = st[1:, cv2.CC_STAT_AREA].astype(float)
        return dict(
            mesh_median_area=float(np.median(areas)),
            mesh_cv_area=float(np.std(areas) / areas.mean()) if areas.mean() else 0.0,
            mesh_max_area_ratio=float(areas.max() / (res * res)),
        )

    def _betweenness_edges(self, G):
        if G.number_of_nodes() <= 2500:
            bc = nx.edge_betweenness_centrality(G, normalized=True)
        else:
            bc = nx.edge_betweenness_centrality(G, k=200, normalized=True, seed=0)
        vals = np.array(list(bc.values()))
        return dict(
            edge_betweenness_max=float(vals.max() if vals.size else 0.0),
            edge_betweenness_gini=float(self._gini(vals)),
        )

    # ==================== Pore Features (18) ====================

    def _pore_features(self, img):
        res = self.canvas_size
        total_px = res * res
        work = img.copy()
        cv2.floodFill(work, None, (0, 0), 128)
        mask = (work == 255).astype(np.uint8)
        cc_n, labels, st, cen = cv2.connectedComponentsWithStats(mask, 8)

        zero = dict(
            # --- original 10 ---
            largest_pore_ratio=0.0, top_area_sum_ratio=0.0,
            top_convexity_min=1.0, top_convexity_mean=1.0, top_circularity_min=1.0,
            big_pore_count=0, total_pore_count=0,
            total_pore_ratio=0.0, center_pore_ratio=0.0, edge_pore_ratio=0.0,
            # --- new 8 ---
            pore_area_cv=0.0, pore_area_skew=0.0, pore_area_kurtosis=0.0,
            pore_area_max_over_mean=1.0, pore_large_area_frac=0.0,
            pore_count_large_frac=0.0, pore_density=0.0, pore_spatial_cv=0.0,
        )
        if cc_n <= 1:
            return zero

        # ---- all pore areas & centroids (label 0 = background) ----
        all_areas = st[1:, cv2.CC_STAT_AREA].astype(float)   # (n_pores,)
        all_cxy   = cen[1:]                                    # (n_pores, 2)
        n_pores   = len(all_areas)

        # ---- NEW: distribution statistics (scale-invariant) ----
        if n_pores >= 2:
            mean_a = float(all_areas.mean())
            std_a  = float(all_areas.std(ddof=0))
            pore_area_cv          = std_a / mean_a if mean_a > 0 else 0.0
            pore_area_skew        = float(sp_skew(all_areas))
            pore_area_kurtosis    = float(sp_kurtosis(all_areas))
            pore_area_max_over_mean = float(all_areas.max() / mean_a) if mean_a > 0 else 1.0
            large_mask = all_areas > 2.0 * mean_a
            total_area_sum = float(all_areas.sum())
            pore_large_area_frac  = float(all_areas[large_mask].sum() / total_area_sum) if total_area_sum > 0 else 0.0
            pore_count_large_frac = float(large_mask.sum() / n_pores)
        else:
            pore_area_cv = 0.0
            pore_area_skew = 0.0
            pore_area_kurtosis = 0.0
            pore_area_max_over_mean = 1.0
            pore_large_area_frac = 0.0
            pore_count_large_frac = 0.0

        pore_density = float(n_pores / total_px)

        # ---- NEW: spatial uniformity (3×3 grid CV) ----
        grid_n = 3
        grid_area = np.zeros((grid_n, grid_n))
        for k in range(n_pores):
            cx, cy = all_cxy[k]
            xi = min(int(cx / res * grid_n), grid_n - 1)
            yi = min(int(cy / res * grid_n), grid_n - 1)
            grid_area[yi, xi] += all_areas[k]
        nz = grid_area[grid_area > 0]
        pore_spatial_cv = float(nz.std() / nz.mean()) if len(nz) > 1 else 0.0

        new_feats = dict(
            pore_area_cv=pore_area_cv,
            pore_area_skew=pore_area_skew,
            pore_area_kurtosis=pore_area_kurtosis,
            pore_area_max_over_mean=pore_area_max_over_mean,
            pore_large_area_frac=pore_large_area_frac,
            pore_count_large_frac=pore_count_large_frac,
            pore_density=pore_density,
            pore_spatial_cv=pore_spatial_cv,
        )

        # ---- original: center / edge split ----
        margin = res * self.edge_margin
        cp, ep = [], []
        for i in range(1, cc_n):
            a = st[i, cv2.CC_STAT_AREA]
            r = a / total_px
            cx, cy = cen[i]
            if margin < cx < res - margin and margin < cy < res - margin:
                cp.append((i, a, r))
            else:
                ep.append((i, a, r))

        tpc = len(cp) + len(ep)
        tpr = (sum(a for _, a, _ in cp) + sum(a for _, a, _ in ep)) / total_px
        cpr = sum(a for _, a, _ in cp) / total_px
        epr = sum(a for _, a, _ in ep) / total_px

        big = sorted([(i, a, r) for i, a, r in cp if r >= self.area_thresh],
                     key=lambda x: x[1], reverse=True)
        bpc = len(big)
        top = big[:self.top_k]
        base = dict(big_pore_count=bpc, total_pore_count=tpc,
                    total_pore_ratio=tpr, center_pore_ratio=cpr, edge_pore_ratio=epr)
        if not top:
            return {**zero, **base, **new_feats}

        def _shape(lid):
            m = (labels == lid).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return 1.0, 1.0
            c = cnts[0]
            ar = cv2.contourArea(c)
            ha = cv2.contourArea(cv2.convexHull(c))
            pe = cv2.arcLength(c, True)
            return (ar / ha if ha else 1.0), (4 * np.pi * ar / pe ** 2 if pe else 1.0)

        shapes = [_shape(i) for i, _, _ in top]
        ars = [r for _, _, r in top]
        return dict(
            largest_pore_ratio=ars[0], top_area_sum_ratio=sum(ars),
            top_convexity_min=min(s[0] for s in shapes),
            top_convexity_mean=float(np.mean([s[0] for s in shapes])),
            top_circularity_min=min(s[1] for s in shapes),
            **base, **new_feats)

    # ==================== Contact Features (42) ====================

    def _contact_features(self, G, id2pt):
        edges = list(G.edges())
        E = len(edges)
        N = G.number_of_nodes()

        node_to_edges = defaultdict(set)
        for i, (u, v) in enumerate(edges):
            node_to_edges[u].add(i)
            node_to_edges[v].add(i)
        adj_pairs = set()
        for eset in node_to_edges.values():
            lst = list(eset)
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    adj_pairs.add(tuple(sorted((lst[i], lst[j]))))

        edge_pixels = [self._get_edge_pixels(id2pt[u], id2pt[v], self.thick)
                       for u, v in edges]
        ep_counts = [len(s) for s in edge_pixels]
        ep_sum = int(np.sum(ep_counts))
        ep_union = set().union(*edge_pixels) if edge_pixels else set()
        ep_union_n = len(ep_union)

        px2e = defaultdict(list)
        for i, pxs in enumerate(edge_pixels):
            for px in pxs:
                px2e[px].append(i)

        raw_olap = set()
        olap = defaultdict(set)
        for px, el in px2e.items():
            if len(el) >= 2:
                raw_olap.add(px)
                for i in range(len(el)):
                    for j in range(i + 1, len(el)):
                        pair = tuple(sorted((el[i], el[j])))
                        if pair not in adj_pairs:
                            olap[pair].add(px)

        olap_all = set()
        ecd = np.zeros(E, dtype=int)
        for (a, b), pxs in olap.items():
            if pxs:
                olap_all.update(pxs)
                ecd[a] += 1
                ecd[b] += 1

        olap_n = len(olap_all)
        raw_n = len(raw_olap)
        pair_n = len(olap)

        cc_st = self._safe_stats(self._cc_sizes_from_pixels(olap_all, self.connectivity))
        pr_st = self._safe_stats([len(p) for p in olap.values()])
        ec_st = self._safe_stats(ecd.tolist())

        cl_len = float(sum(math.dist(id2pt[u], id2pt[v]) for u, v in edges))
        ca = self.canvas_size ** 2
        ewc = int((ecd > 0).sum())
        ol_approx = olap_n / max(self.thick, 1)

        return {
            "contact_thick": int(self.thick),
            "contact_canvas_size": int(self.canvas_size),
            "contact_nodes": N,
            "contact_edges": E,
            "contact_edge_pixel_union_count": ep_union_n,
            "contact_edge_pixel_sum": ep_sum,
            "contact_raw_overlap_pixel_count": raw_n,
            "contact_overlap_pixel_count": olap_n,
            "contact_overlap_pair_count": pair_n,
            "contact_overlap_pairs_per_edge": pair_n / E if E else 0.0,
            "contact_edges_with_contact_count": ewc,
            "contact_edges_with_contact_ratio": ewc / E if E else 0.0,
            "contact_overlap_pixel_ratio_union": olap_n / ep_union_n if ep_union_n else 0.0,
            "contact_raw_overlap_pixel_ratio_union": raw_n / ep_union_n if ep_union_n else 0.0,
            "contact_overlap_pixel_ratio_canvas": olap_n / ca if ca else 0.0,
            "contact_overlap_length_px_approx": ol_approx,
            "contact_overlap_length_ratio_centerline": ol_approx / cl_len if cl_len else 0.0,
            "contact_centerline_length_px": cl_len,
            "contact_overlap_pair_size_sum": pr_st["sum"],
            "contact_overlap_pair_size_mean": pr_st["mean"],
            "contact_overlap_pair_size_median": pr_st["median"],
            "contact_overlap_pair_size_max": pr_st["max"],
            "contact_overlap_pair_size_std": pr_st["std"],
            "contact_overlap_pair_size_q75": pr_st["q75"],
            "contact_overlap_pair_size_q90": pr_st["q90"],
            "contact_overlap_pair_size_q95": pr_st["q95"],
            "contact_overlap_cc_count": cc_st["count"],
            "contact_overlap_cc_size_sum": cc_st["sum"],
            "contact_overlap_cc_size_mean": cc_st["mean"],
            "contact_overlap_cc_size_median": cc_st["median"],
            "contact_overlap_cc_size_max": cc_st["max"],
            "contact_overlap_cc_size_std": cc_st["std"],
            "contact_overlap_cc_size_q75": cc_st["q75"],
            "contact_overlap_cc_size_q90": cc_st["q90"],
            "contact_overlap_cc_size_q95": cc_st["q95"],
            "contact_edge_contact_degree_mean": ec_st["mean"],
            "contact_edge_contact_degree_median": ec_st["median"],
            "contact_edge_contact_degree_max": ec_st["max"],
            "contact_edge_contact_degree_std": ec_st["std"],
            "contact_edge_contact_degree_q75": ec_st["q75"],
            "contact_edge_contact_degree_q90": ec_st["q90"],
            "contact_edge_contact_degree_q95": ec_st["q95"],
        }

    # ==================== Public API ====================

    def extract(self, key: str) -> dict:
        fp = self.root / f"{key}.json"
        if not fp.exists():
            return {}
        G = self._load_graph(fp)
        img, id2pt = self._render_image(G)
        feat = {}
        feat.update(self._basic_size(G))
        feat.update(self._degree_stats(G))
        feat.update(self._orientation_stats(G))
        feat.update(self._spatial_moments(G))
        feat.update(self._path_connectivity(G))
        feat.update(self._boundary_fractal(G))
        feat.update(self._redundancy_kcore(G))
        feat.update(self._cycle_features(G))
        feat.update(self._vertical_shortestness(G))
        feat.update(self._mesh_holes(img))
        feat.update(self._betweenness_edges(G))
        feat.update(self._pore_features(img))
        feat.update(self._contact_features(G, id2pt))
        return {"key": key, **feat}

    def batch(self, keys):
        recs, missing = [], []
        for k in tqdm(keys, desc="Extracting 94-d features"):
            d = self.extract(k)
            if d:
                recs.append(d)
            else:
                missing.append(k)
        return pd.DataFrame(recs), missing


# ==================== CLI ====================

if __name__ == "__main__":
    GRAPH_ROOT = Path(r"D:\CODE\Abaqus_batch_test\Dataset\Weld_Graph_Data")
    CSV_FORCE = Path(r"D:\CODE\Abaqus_batch_test\data_analysis\combined_forces_cleaned.csv")
    OUT_CSV = GRAPH_ROOT / "features_94d.csv"

    ext = GraphFeatureExtractor(
        GRAPH_ROOT, canvas_size=1024, thick=9,
        edge_margin=0.12, top_k=3, area_thresh=0.005, connectivity=8,
    )
    print(f"[Config] canvas={ext.canvas_size}  thick={ext.thick}  "
          f"margin={ext.edge_margin}  top_k={ext.top_k}  area_thresh={ext.area_thresh}")

    force_df = pd.read_csv(CSV_FORCE)
    sample_ids = list(force_df.columns[1:])
    print(f"[Info] samples: {len(sample_ids)}")

    df, miss = ext.batch(sample_ids)
    print(f"[Done] extracted: {len(df)} | missing: {len(miss)}")

    y_row = force_df.loc[force_df.Time == 1].squeeze()
    df["max_load"] = df["key"].map(y_row.to_dict())

    col_order = ["key"] + GraphFeatureExtractor.FEATURE_COLS + ["max_load"]
    df = df[[c for c in col_order if c in df.columns]]

    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    print(f"[Save] {OUT_CSV}  shape={df.shape}")
    print(f"[Time] {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    if miss:
        print(f"[Warn] missing (first 10): {miss[:10]}")
    print("\n[Preview]")
    print(df.head(3).to_string())