"""Preprocessing, PCA, K selection and clustering (study procedure).

Preprocessing per column: median imputation, log1p of non-negative columns
with |skew| > 1.5, winsorisation at the 0.5 / 99.5 percentiles,
standardisation, equal weights. Whitened PCA (5 components). K-means
(n_init 10, seed 42) for K in 5..15.

K-selection rule: K is admissible when (a) every type holds at least
`min_type_size` blocks and (b) K is not below the inertia elbow (kneed).
Among admissible K the one with the highest combined score (mean of the
min-max normalised silhouette, Calinski-Harabasz and inverted
Davies-Bouldin) is selected. When no K meets both conditions, the study
code fell back to the best combined score after relaxing one condition;
this module reproduces that fallback but REPORTS it (`status`), so a
fallback partition is never presented as a rule result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .params import Params


def prepare_matrix(df: pd.DataFrame, log_skew: float = 1.5) -> np.ndarray:
    x = df.copy()
    for col in list(x.columns):
        s = x[col]
        if s.isna().all():
            x = x.drop(columns=col)
            continue
        s = s.fillna(s.median())
        if (s.min() >= 0) and abs(s.skew()) > log_skew:
            s = np.log1p(s)
        x[col] = s.clip(s.quantile(0.005), s.quantile(0.995))
    return StandardScaler().fit_transform(x.values)


def pca_space(feats: pd.DataFrame, p: Params):
    x = prepare_matrix(feats)
    k = max(1, min(p.pca_components, x.shape[0] - 1, x.shape[1]))
    pca = PCA(n_components=k, whiten=p.pca_whiten, random_state=p.random_state)
    return pca.fit_transform(x), pca


def kmeans(z, k, p: Params, seed=None):
    return KMeans(n_clusters=k, n_init=p.kmeans_n_init,
                  random_state=p.random_state if seed is None else seed).fit_predict(z)


def _noise_silhouette(n, dims, k, reps=3):
    rng = np.random.default_rng(k * 1000 + dims)
    vals = []
    for r in range(reps):
        zn = rng.normal(size=(n, dims))
        vals.append(silhouette_score(zn, KMeans(k, n_init=5, random_state=r).fit_predict(zn)))
    return float(np.mean(vals))


def select_k(z, p: Params):
    """Returns (labels, sweep table, status dict)."""
    n = len(z)
    if n < p.min_type_size:
        raise RuntimeError(f"only {n} blocks - too few to cluster")
    ks = [k for k in p.k_range if k <= n - 1]
    rows, fits = [], {}
    for k in ks:
        est = KMeans(n_clusters=k, n_init=p.kmeans_n_init, random_state=p.random_state)
        lab = est.fit_predict(z)
        sizes = np.bincount(lab, minlength=k)
        sil = silhouette_score(z, lab)
        noise = _noise_silhouette(n, z.shape[1], k)
        rows.append({"k": k, "silhouette": sil, "noise_silhouette": noise,
                     "silhouette_gap": sil - noise,
                     "calinski": calinski_harabasz_score(z, lab),
                     "davies": davies_bouldin_score(z, lab), "inertia": est.inertia_,
                     "min_type_size": int(sizes.min()),
                     "size_ok": bool(sizes.min() >= p.min_type_size)})
        fits[k] = lab
    sw = pd.DataFrame(rows)

    def nm(s):
        r = s.max() - s.min()
        return (s - s.min()) / r if r > 0 else s * 0
    sw["combined"] = (nm(sw["silhouette"]) + nm(sw["calinski"]) + (1 - nm(sw["davies"]))) / 3
    try:
        from kneed import KneeLocator
        elbow = KneeLocator(sw["k"], sw["inertia"], curve="convex", direction="decreasing").knee
    except Exception:                                            # noqa: BLE001
        elbow = None
    sw["elbow_k"] = elbow
    sw["above_elbow"] = True if elbow is None else sw["k"] >= int(elbow)
    sw["admissible"] = sw["size_ok"] & sw["above_elbow"]
    if sw["admissible"].any():
        cand, status = sw[sw["admissible"]], "rule"
        reason = "K meets both conditions (type size and elbow)"
    else:
        # the study code's fallback, made explicit
        size_ok = sw[sw["size_ok"]]
        if len(size_ok):
            cand, status = size_ok, "fallback"
            reason = ("no K meets both conditions; elbow condition relaxed "
                      "(best combined score among K whose types are large enough)")
        else:
            above = sw[sw["above_elbow"]]
            cand, status = (above if len(above) else sw), "fallback"
            reason = ("no K meets both conditions; type-size condition relaxed "
                      "(best combined score at or above the elbow)")
    k = int(cand.loc[cand["combined"].idxmax(), "k"])
    sw["selected"] = sw["k"] == k
    status_d = {"selected_k": k, "status": status, "reason": reason,
                "elbow_k": None if elbow is None else int(elbow),
                "admissible_k": [int(x) for x in sw.loc[sw["admissible"], "k"]]}
    return fits[k], sw, status_d


def stability(feats, k, p: Params, reps=None):
    """Median pairwise ARI of refits on random subsamples (study procedure:
    100 x 80 %, full refit of preprocessing, PCA and K-means)."""
    from sklearn.metrics import adjusted_rand_score
    reps = reps or p.stability_reps
    n = len(feats)
    m = int(round(p.stability_fraction * n))
    F = np.full((reps, n), -1, dtype=int)
    for i in range(reps):
        idx = np.random.RandomState(p.random_state + i).choice(n, size=m, replace=False)
        z, _ = pca_space(feats.iloc[idx], p)
        F[i, idx] = kmeans(z, k, p, seed=p.random_state + i)
    a = [adjusted_rand_score(F[i][(F[i] >= 0) & (F[j] >= 0)], F[j][(F[i] >= 0) & (F[j] >= 0)])
         for i in range(reps) for j in range(i + 1, reps)]
    return {"reps": reps, "median_ari": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75))}
