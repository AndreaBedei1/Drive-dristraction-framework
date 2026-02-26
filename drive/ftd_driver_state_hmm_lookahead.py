"""
Driver-state-only HMM Fitness-to-Drive impairment pipeline.

Goal:
Predict P(error in next T seconds) using only driver-state signals around
distraction windows, with hangover horizon H and no data leakage.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ftd_driver_state_lookahead import (
    FEATURE_COLS,
    RANDOM_SEED,
    add_causal_session_features,
    apply_feature_postprocess,
    apply_imputation_stats,
    bootstrap_ci,
    build_baselines,
    build_encoding_map,
    build_lookups,
    eval_metrics,
    fit_feature_postprocess,
    fit_label_encoders,
    fit_train_imputation_stats,
    generate_samples,
    load_data,
    parse_int_list,
    plot_calibration,
    plot_ht_heatmaps,
    plot_risk_profile,
    plot_roc_pr,
    quick_config_search,
    resolve_xgb_device,
    run_integrity_checks,
    select_threshold_f1,
    split_users,
)


LOG = logging.getLogger("ftd_driver_state_hmm")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def logsumexp_np(a: np.ndarray, axis=None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    den = np.clip(mat.sum(axis=1, keepdims=True), eps, None)
    return mat / den


class GaussianHMMDiagNumpy:
    def __init__(
        self,
        n_states: int = 3,
        max_iter: int = 45,
        tol: float = 1e-3,
        min_covar: float = 1e-3,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.n_states = int(n_states)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_covar = float(min_covar)
        self.random_state = int(random_state)
        self.pi_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.train_loglik_: List[float] = []

    def _initialize(self, X: np.ndarray) -> None:
        n, d = X.shape
        if n < self.n_states:
            raise RuntimeError(f"Need at least {self.n_states} samples to initialize HMM")

        km = MiniBatchKMeans(
            n_clusters=self.n_states,
            random_state=self.random_state,
            batch_size=min(4096, n),
            n_init=10,
        )
        labels = km.fit_predict(X)
        means = km.cluster_centers_.astype(float)

        gvar = np.var(X, axis=0) + self.min_covar
        vars_ = np.zeros((self.n_states, d), dtype=float)
        for k in range(self.n_states):
            mask = labels == k
            if np.any(mask):
                vars_[k] = np.var(X[mask], axis=0) + self.min_covar
            else:
                vars_[k] = gvar

        self.means_ = means
        self.vars_ = np.clip(vars_, self.min_covar, None)
        self.pi_ = np.full(self.n_states, 1.0 / self.n_states, dtype=float)
        off = 0.1 / max(self.n_states - 1, 1)
        A = np.full((self.n_states, self.n_states), off, dtype=float)
        np.fill_diagonal(A, 0.9)
        self.A_ = normalize_rows(A)

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        assert self.means_ is not None and self.vars_ is not None
        diff = X[:, None, :] - self.means_[None, :, :]
        term = np.sum((diff * diff) / self.vars_[None, :, :], axis=2)
        log_det = np.sum(np.log(2.0 * np.pi * self.vars_), axis=1)
        return -0.5 * (term + log_det[None, :])

    def _forward_backward(self, X_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        assert self.pi_ is not None and self.A_ is not None
        T = X_seq.shape[0]
        K = self.n_states
        log_pi = np.log(np.clip(self.pi_, 1e-12, None))
        log_A = np.log(np.clip(self.A_, 1e-12, None))
        log_b = self._log_emission(X_seq)

        alpha = np.empty((T, K), dtype=float)
        alpha[0] = log_pi + log_b[0]
        for t in range(1, T):
            alpha[t] = log_b[t] + logsumexp_np(alpha[t - 1][:, None] + log_A, axis=0)

        loglik = float(logsumexp_np(alpha[-1], axis=0))

        beta = np.zeros((T, K), dtype=float)
        for t in range(T - 2, -1, -1):
            beta[t] = logsumexp_np(log_A + log_b[t + 1][None, :] + beta[t + 1][None, :], axis=1)

        log_gamma = alpha + beta - loglik
        gamma = np.exp(log_gamma)
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), 1e-12, None)

        xi_sum = np.zeros((K, K), dtype=float)
        for t in range(T - 1):
            log_xi = alpha[t][:, None] + log_A + log_b[t + 1][None, :] + beta[t + 1][None, :] - loglik
            xi_sum += np.exp(log_xi)

        return gamma, xi_sum, loglik

    def fit(self, X: np.ndarray, bounds: Sequence[Tuple[int, int]]) -> "GaussianHMMDiagNumpy":
        X = np.asarray(X, dtype=float)
        self._initialize(X)
        assert self.pi_ is not None and self.A_ is not None
        assert self.means_ is not None and self.vars_ is not None

        prev_ll = None
        for it in range(self.max_iter):
            pi_acc = np.zeros(self.n_states, dtype=float)
            A_acc = np.zeros((self.n_states, self.n_states), dtype=float)
            gamma_sum = np.zeros(self.n_states, dtype=float)
            mean_num = np.zeros_like(self.means_)
            sq_num = np.zeros_like(self.vars_)
            total_ll = 0.0

            for s, e in bounds:
                if e - s <= 0:
                    continue
                X_seq = X[s:e]
                gamma, xi, ll = self._forward_backward(X_seq)
                total_ll += ll
                pi_acc += gamma[0]
                A_acc += xi
                gamma_sum += gamma.sum(axis=0)
                mean_num += gamma.T @ X_seq
                sq_num += gamma.T @ (X_seq * X_seq)

            gamma_safe = np.clip(gamma_sum, 1e-8, None)
            new_means = self.means_.copy()
            active = gamma_sum > 1e-8
            new_means[active] = mean_num[active] / gamma_safe[active, None]

            new_vars = self.vars_.copy()
            centered_second = sq_num / gamma_safe[:, None] - new_means * new_means
            centered_second = np.clip(centered_second, self.min_covar, None)
            new_vars[active] = centered_second[active]

            self.pi_ = pi_acc + 1e-3
            self.pi_ /= np.clip(self.pi_.sum(), 1e-12, None)
            self.A_ = normalize_rows(A_acc + 1e-3)
            self.means_ = new_means
            self.vars_ = new_vars

            self.train_loglik_.append(float(total_ll))
            if prev_ll is not None:
                delta = float(total_ll - prev_ll)
                LOG.info("hmm_em iter=%s loglik=%.3f delta=%.6f", it + 1, total_ll, delta)
                if abs(delta) <= self.tol * (abs(prev_ll) + 1.0):
                    break
            else:
                LOG.info("hmm_em iter=%s loglik=%.3f", it + 1, total_ll)
            prev_ll = total_ll

        return self

    def predict_posteriors(self, X: np.ndarray, bounds: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
        X = np.asarray(X, dtype=float)
        gamma_all = np.zeros((X.shape[0], self.n_states), dtype=float)
        total_ll = 0.0
        for s, e in bounds:
            if e - s <= 0:
                continue
            gamma, _, ll = self._forward_backward(X[s:e])
            gamma_all[s:e] = gamma
            total_ll += ll
        return gamma_all, float(total_ll)

    def to_dict(self) -> Dict:
        return {
            "n_states": int(self.n_states),
            "pi": np.asarray(self.pi_, dtype=float),
            "A": np.asarray(self.A_, dtype=float),
            "means": np.asarray(self.means_, dtype=float),
            "vars": np.asarray(self.vars_, dtype=float),
            "train_loglik": [float(x) for x in self.train_loglik_],
        }


class GaussianHMMDiagTorch:
    def __init__(
        self,
        n_states: int = 3,
        max_iter: int = 45,
        tol: float = 1e-3,
        min_covar: float = 1e-3,
        random_state: int = RANDOM_SEED,
        device: str = "cuda",
    ) -> None:
        import torch

        self.torch = torch
        self.n_states = int(n_states)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_covar = float(min_covar)
        self.random_state = int(random_state)
        self.device = str(device)
        self.pi_ = None
        self.A_ = None
        self.means_ = None
        self.vars_ = None
        self.train_loglik_: List[float] = []

    def _initialize(self, X: np.ndarray) -> None:
        t = self.torch
        n, d = X.shape
        if n < self.n_states:
            raise RuntimeError(f"Need at least {self.n_states} samples to initialize HMM")
        km = MiniBatchKMeans(
            n_clusters=self.n_states,
            random_state=self.random_state,
            batch_size=min(4096, n),
            n_init=10,
        )
        labels = km.fit_predict(X)
        means = km.cluster_centers_.astype(np.float64)
        gvar = np.var(X, axis=0) + self.min_covar
        vars_ = np.zeros((self.n_states, d), dtype=np.float64)
        for k in range(self.n_states):
            mask = labels == k
            if np.any(mask):
                vars_[k] = np.var(X[mask], axis=0) + self.min_covar
            else:
                vars_[k] = gvar

        self.means_ = t.tensor(means, dtype=t.float64, device=self.device)
        self.vars_ = t.tensor(np.clip(vars_, self.min_covar, None), dtype=t.float64, device=self.device)
        self.pi_ = t.full((self.n_states,), 1.0 / self.n_states, dtype=t.float64, device=self.device)
        off = 0.1 / max(self.n_states - 1, 1)
        A = np.full((self.n_states, self.n_states), off, dtype=np.float64)
        np.fill_diagonal(A, 0.9)
        A = normalize_rows(A)
        self.A_ = t.tensor(A, dtype=t.float64, device=self.device)

    def _log_emission(self, X_seq):
        t = self.torch
        diff = X_seq[:, None, :] - self.means_[None, :, :]
        term = t.sum((diff * diff) / self.vars_[None, :, :], dim=2)
        log_det = t.sum(t.log(2.0 * np.pi * self.vars_), dim=1)
        return -0.5 * (term + log_det[None, :])

    def _forward_backward(self, X_seq):
        t = self.torch
        T = int(X_seq.shape[0])
        K = self.n_states
        log_pi = t.log(t.clamp(self.pi_, min=1e-12))
        log_A = t.log(t.clamp(self.A_, min=1e-12))
        log_b = self._log_emission(X_seq)

        alpha = t.empty((T, K), dtype=t.float64, device=self.device)
        alpha[0] = log_pi + log_b[0]
        for i in range(1, T):
            alpha[i] = log_b[i] + t.logsumexp(alpha[i - 1].unsqueeze(1) + log_A, dim=0)

        loglik = t.logsumexp(alpha[-1], dim=0)

        beta = t.zeros((T, K), dtype=t.float64, device=self.device)
        for i in range(T - 2, -1, -1):
            beta[i] = t.logsumexp(log_A + log_b[i + 1].unsqueeze(0) + beta[i + 1].unsqueeze(0), dim=1)

        log_gamma = alpha + beta - loglik
        gamma = t.exp(log_gamma)
        gamma = gamma / t.clamp(gamma.sum(dim=1, keepdim=True), min=1e-12)

        xi_sum = t.zeros((K, K), dtype=t.float64, device=self.device)
        for i in range(T - 1):
            log_xi = alpha[i].unsqueeze(1) + log_A + log_b[i + 1].unsqueeze(0) + beta[i + 1].unsqueeze(0) - loglik
            xi_sum += t.exp(log_xi)

        return gamma, xi_sum, float(loglik.item())

    def fit(self, X: np.ndarray, bounds: Sequence[Tuple[int, int]]):
        t = self.torch
        X_np = np.asarray(X, dtype=np.float64)
        self._initialize(X_np)
        X_all = t.tensor(X_np, dtype=t.float64, device=self.device)

        prev_ll = None
        for it in range(self.max_iter):
            pi_acc = t.zeros((self.n_states,), dtype=t.float64, device=self.device)
            A_acc = t.zeros((self.n_states, self.n_states), dtype=t.float64, device=self.device)
            gamma_sum = t.zeros((self.n_states,), dtype=t.float64, device=self.device)
            mean_num = t.zeros_like(self.means_)
            sq_num = t.zeros_like(self.vars_)
            total_ll = 0.0

            for s, e in bounds:
                if e - s <= 0:
                    continue
                X_seq = X_all[s:e]
                gamma, xi, ll = self._forward_backward(X_seq)
                total_ll += ll
                pi_acc += gamma[0]
                A_acc += xi
                gamma_sum += gamma.sum(dim=0)
                mean_num += gamma.t() @ X_seq
                sq_num += gamma.t() @ (X_seq * X_seq)

            gamma_safe = t.clamp(gamma_sum, min=1e-8)
            new_means = self.means_.clone()
            active = gamma_sum > 1e-8
            new_means[active] = mean_num[active] / gamma_safe[active].unsqueeze(1)

            centered_second = sq_num / gamma_safe.unsqueeze(1) - new_means * new_means
            centered_second = t.clamp(centered_second, min=self.min_covar)
            new_vars = self.vars_.clone()
            new_vars[active] = centered_second[active]

            self.pi_ = pi_acc + 1e-3
            self.pi_ = self.pi_ / t.clamp(self.pi_.sum(), min=1e-12)
            self.A_ = A_acc + 1e-3
            self.A_ = self.A_ / t.clamp(self.A_.sum(dim=1, keepdim=True), min=1e-12)
            self.means_ = new_means
            self.vars_ = new_vars

            self.train_loglik_.append(float(total_ll))
            if prev_ll is not None:
                delta = float(total_ll - prev_ll)
                LOG.info("hmm_em iter=%s loglik=%.3f delta=%.6f", it + 1, total_ll, delta)
                if abs(delta) <= self.tol * (abs(prev_ll) + 1.0):
                    break
            else:
                LOG.info("hmm_em iter=%s loglik=%.3f", it + 1, total_ll)
            prev_ll = total_ll
        return self

    def predict_posteriors(self, X: np.ndarray, bounds: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, float]:
        t = self.torch
        X_all = t.tensor(np.asarray(X, dtype=np.float64), dtype=t.float64, device=self.device)
        gamma_all = t.zeros((X_all.shape[0], self.n_states), dtype=t.float64, device=self.device)
        total_ll = 0.0
        for s, e in bounds:
            if e - s <= 0:
                continue
            gamma, _, ll = self._forward_backward(X_all[s:e])
            gamma_all[s:e] = gamma
            total_ll += ll
        return gamma_all.detach().cpu().numpy().astype(float), float(total_ll)

    def to_dict(self) -> Dict:
        return {
            "n_states": int(self.n_states),
            "pi": self.pi_.detach().cpu().numpy().astype(float),
            "A": self.A_.detach().cpu().numpy().astype(float),
            "means": self.means_.detach().cpu().numpy().astype(float),
            "vars": self.vars_.detach().cpu().numpy().astype(float),
            "train_loglik": [float(x) for x in self.train_loglik_],
        }


def sort_for_hmm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_ord"] = np.arange(len(out))
    out = out.sort_values(["user_id", "run_id", "sample_ts", "_ord"]).reset_index(drop=True)
    return out.drop(columns=["_ord"])


def sequence_bounds(df: pd.DataFrame) -> List[Tuple[int, int]]:
    bounds: List[Tuple[int, int]] = []
    for _, grp in df.groupby(["user_id", "run_id"], sort=False):
        s = int(grp.index.min())
        e = int(grp.index.max()) + 1
        if e > s:
            bounds.append((s, e))
    return bounds


def torch_cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_hmm_runtime(requested: str) -> Dict[str, str]:
    req = str(requested).strip().lower()
    if req not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported HMM device '{requested}'. Use auto|cpu|cuda.")

    has_torch = importlib.util.find_spec("torch") is not None
    has_cuda = torch_cuda_available()

    if req == "cuda":
        if has_torch and has_cuda:
            return {"requested": req, "resolved": "cuda", "backend": "torch"}
        LOG.warning("HMM CUDA requested but torch/cuda is unavailable. Falling back to CPU numpy backend.")
        return {"requested": req, "resolved": "cpu", "backend": "numpy"}
    if req == "cpu":
        return {"requested": req, "resolved": "cpu", "backend": "numpy"}

    if has_torch and has_cuda:
        return {"requested": req, "resolved": "cuda", "backend": "torch"}
    return {"requested": req, "resolved": "cpu", "backend": "numpy"}


def derive_state_profile(gamma: np.ndarray, y_true: np.ndarray) -> Dict:
    mass = np.clip(gamma.sum(axis=0), 1e-8, None)
    risk = (gamma.T @ y_true) / mass
    order = np.argsort(risk)
    labels = ["soft impairment", "mild impairment", "hard impairment"]
    by_state = {int(order[i]): labels[i] for i in range(len(order))}
    return {
        "state_risk": [float(x) for x in risk],
        "state_mass": [float(x) for x in mass],
        "risk_order_low_to_high": [int(x) for x in order],
        "state_semantics": by_state,
    }


def apply_score_calibration(name: str, model, prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=float)
    if name == "none":
        return np.clip(p, 1e-8, 1.0 - 1e-8)
    if name == "sigmoid":
        return np.clip(model.predict_proba(p.reshape(-1, 1))[:, 1], 1e-8, 1.0 - 1e-8)
    if name == "isotonic":
        return np.clip(model.predict(p), 1e-8, 1.0 - 1e-8)
    raise ValueError(f"Unknown calibration '{name}'")


def calibrate_scores(y_cal: np.ndarray, prob_raw_cal: np.ndarray):
    candidates: List[Tuple[str, object, np.ndarray]] = []
    p0 = np.clip(np.asarray(prob_raw_cal, dtype=float), 1e-8, 1.0 - 1e-8)
    candidates.append(("none", None, p0))

    try:
        lr = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_SEED,
        )
        lr.fit(p0.reshape(-1, 1), y_cal)
        candidates.append(("sigmoid", lr, apply_score_calibration("sigmoid", lr, p0)))
    except Exception as exc:
        LOG.warning("sigmoid calibration failed: %s", exc)

    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p0, y_cal)
        candidates.append(("isotonic", iso, apply_score_calibration("isotonic", iso, p0)))
    except Exception as exc:
        LOG.warning("isotonic calibration failed: %s", exc)

    best = None
    best_key = None
    records = []
    for name, model, p in candidates:
        ap = float(average_precision_score(y_cal, p))
        br = float(brier_score_loss(y_cal, p))
        records.append({"calibration": name, "ap_cal": ap, "brier_cal": br})
        key = (ap, -br)
        if best is None or key > best_key:
            best = (name, model, p)
            best_key = key

    assert best is not None
    return best[0], best[1], best[2], records


def plot_hmm_feature_importance(
    means: np.ndarray,
    state_risk: np.ndarray,
    out_path: Path,
) -> None:
    if means.ndim != 2 or means.shape[1] != len(FEATURE_COLS):
        return
    order = np.argsort(state_risk)
    if len(order) < 3:
        return
    soft = means[order[0]]
    mild = means[order[1]]
    hard = means[order[2]]
    importance = np.abs(hard - soft) + 0.5 * np.abs(hard - mild) + 0.5 * np.abs(mild - soft)

    idx = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(len(idx)), importance[idx])
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels([FEATURE_COLS[i] for i in idx])
    ax.invert_yaxis()
    ax.set_xlabel("State-separation importance (scaled units)")
    ax.set_title("HMM feature importance proxy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_pipeline(args) -> Dict:
    d_raw, e_raw, eb_raw, db_raw = load_data(args.data_path)
    run_integrity_checks(d_raw, e_raw)

    users = sorted(set(d_raw["user_id"].unique()) | set(e_raw["user_id"].unique()))
    train_users, cal_users, test_users = split_users(users, seed=args.seed)
    LOG.info("users: train=%s cal=%s test=%s", len(train_users), len(cal_users), len(test_users))

    impute_stats = fit_train_imputation_stats(train_users, d_raw)
    d = apply_imputation_stats(d_raw, impute_stats)

    le_pred, le_emo = fit_label_encoders(train_users, d, e_raw)
    pred_enc = build_encoding_map(le_pred)
    emo_enc = build_encoding_map(le_emo)
    baselines = build_baselines(train_users, eb_raw, db_raw)

    d_train = d[d["user_id"].isin(set(train_users))]
    global_model_prob = float(d_train[["model_prob_start", "model_prob_end"]].stack().median()) if len(d_train) else 0.5
    global_emotion_prob = float(d_train[["emotion_prob_start", "emotion_prob_end"]].stack().median()) if len(d_train) else 0.5
    if not np.isfinite(global_model_prob):
        global_model_prob = 0.5
    if not np.isfinite(global_emotion_prob):
        global_emotion_prob = 0.5

    wbs, webs, errs = build_lookups(d, e_raw, users)

    # Keep the same H/T search protocol as the XGB pipeline for direct comparability.
    best_h, best_t, search_df = quick_config_search(
        H_vals=args.h_values,
        T_vals=args.t_values,
        train_users=train_users,
        wbs=wbs,
        webs=webs,
        errs=errs,
        baselines=baselines,
        pred_enc=pred_enc,
        emo_enc=emo_enc,
        global_model_prob=global_model_prob,
        global_emotion_prob=global_emotion_prob,
        xgb_device=args.xgb_device_resolved,
        xgb_n_jobs=args.xgb_n_jobs,
        tolerance_ap=args.t_parsimony_tol,
    )

    df_tr = generate_samples(best_h, best_t, train_users, wbs, webs, errs, baselines, pred_enc, emo_enc, global_model_prob, global_emotion_prob)
    df_ca = generate_samples(best_h, best_t, cal_users, wbs, webs, errs, baselines, pred_enc, emo_enc, global_model_prob, global_emotion_prob)
    df_te = generate_samples(best_h, best_t, test_users, wbs, webs, errs, baselines, pred_enc, emo_enc, global_model_prob, global_emotion_prob)
    df_tr = add_causal_session_features(df_tr)
    df_ca = add_causal_session_features(df_ca)
    df_te = add_causal_session_features(df_te)

    for name, df in [("train", df_tr), ("cal", df_ca), ("test", df_te)]:
        if df.empty or df["target"].nunique() < 2:
            raise RuntimeError(f"{name} split has insufficient data/class diversity.")
        LOG.info("%s samples=%s pos_rate=%.4f", name, len(df), float(df["target"].mean()))

    pp = fit_feature_postprocess(df_tr)
    df_tr = apply_feature_postprocess(df_tr, pp)
    df_ca = apply_feature_postprocess(df_ca, pp)
    df_te = apply_feature_postprocess(df_te, pp)

    df_tr = sort_for_hmm(df_tr)
    df_ca = sort_for_hmm(df_ca)
    df_te = sort_for_hmm(df_te)

    X_tr = df_tr[FEATURE_COLS].values.astype(np.float64)
    y_tr = df_tr["target"].values.astype(int)
    X_ca = df_ca[FEATURE_COLS].values.astype(np.float64)
    y_ca = df_ca["target"].values.astype(int)
    X_te = df_te[FEATURE_COLS].values.astype(np.float64)
    y_te = df_te["target"].values.astype(int)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ca = scaler.transform(X_ca)
    X_te = scaler.transform(X_te)

    b_tr = sequence_bounds(df_tr)
    b_ca = sequence_bounds(df_ca)
    b_te = sequence_bounds(df_te)

    if args.hmm_runtime["backend"] == "torch":
        hmm = GaussianHMMDiagTorch(
            n_states=args.hmm_states,
            max_iter=args.hmm_max_iter,
            tol=args.hmm_tol,
            min_covar=args.hmm_min_covar,
            random_state=args.seed,
            device=args.hmm_runtime["resolved"],
        )
    else:
        hmm = GaussianHMMDiagNumpy(
            n_states=args.hmm_states,
            max_iter=args.hmm_max_iter,
            tol=args.hmm_tol,
            min_covar=args.hmm_min_covar,
            random_state=args.seed,
        )

    hmm.fit(X_tr, b_tr)
    gamma_tr, ll_tr = hmm.predict_posteriors(X_tr, b_tr)
    gamma_ca, ll_ca = hmm.predict_posteriors(X_ca, b_ca)
    gamma_te, ll_te = hmm.predict_posteriors(X_te, b_te)
    LOG.info("hmm ll: train=%.2f cal=%.2f test=%.2f", ll_tr, ll_ca, ll_te)

    state_profile = derive_state_profile(gamma_tr, y_tr)
    state_risk = np.asarray(state_profile["state_risk"], dtype=float)
    p_ca_raw = np.clip(gamma_ca @ state_risk, 1e-8, 1.0 - 1e-8)
    p_te_raw = np.clip(gamma_te @ state_risk, 1e-8, 1.0 - 1e-8)

    cal_name, cal_model, p_ca, cal_records = calibrate_scores(y_ca, p_ca_raw)
    p_te = apply_score_calibration(cal_name, cal_model, p_te_raw)
    threshold = select_threshold_f1(y_ca, p_ca)

    metrics_test = eval_metrics(y_te, p_te, threshold)
    ci_ap = bootstrap_ci(y_te, p_te, average_precision_score, n_boot=args.bootstrap)
    ci_roc = bootstrap_ci(y_te, p_te, roc_auc_score, n_boot=args.bootstrap)
    ci_brier = bootstrap_ci(y_te, p_te, brier_score_loss, n_boot=args.bootstrap)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_ht_heatmaps(search_df, best_h, best_t, output_dir / "ht_search_heatmaps.png")
    plot_roc_pr(y_te, p_te, output_dir / "roc_pr_curves.png", f"(H={best_h}, T={best_t})")
    plot_calibration(y_te, p_te_raw, p_te, output_dir / "calibration_curve.png", f"(H={best_h}, T={best_t})")
    plot_risk_profile(df_te, p_te, best_h, best_t, output_dir / "risk_profile.png")
    plot_hmm_feature_importance(
        means=np.asarray(hmm.to_dict()["means"], dtype=float),
        state_risk=state_risk,
        out_path=output_dir / "feature_importance.png",
    )

    artifact = {
        "model_family": "gaussian_hmm_diag",
        "hmm": hmm.to_dict(),
        "hmm_runtime": args.hmm_runtime,
        "state_profile": state_profile,
        "calibration": {"name": cal_name, "model": cal_model},
        "scaler": scaler,
        "feature_postprocess": pp,
        "feature_cols": FEATURE_COLS,
        "best_config": {"H": int(best_h), "T": int(best_t)},
        "threshold": float(threshold),
        "label_encoders": {"model_pred": le_pred, "emotion": le_emo},
        "encoding_maps": {"model_pred": pred_enc, "emotion": emo_enc},
        "impute_stats": impute_stats,
        "baselines": baselines,
        "global_driver_state_defaults": {
            "model_prob": float(global_model_prob),
            "emotion_prob": float(global_emotion_prob),
        },
    }
    joblib.dump(artifact, output_dir / "driver_state_hmm_model.joblib")

    search_records = []
    for rec in search_df.to_dict(orient="records"):
        out_rec = {}
        for k, v in rec.items():
            if isinstance(v, (np.floating, float)):
                out_rec[k] = float(v) if np.isfinite(v) else None
            elif isinstance(v, (np.integer, int)):
                out_rec[k] = int(v)
            else:
                out_rec[k] = v
        search_records.append(out_rec)

    result = {
        "constraints": {
            "driver_state_only": True,
            "leakage_controls": [
                "user-level train/cal/test split",
                "train-only imputation",
                "train-only encoders",
                "train-only feature clipping",
                "H/T search on train only",
                "calibration + threshold on calibration only",
                "single locked test evaluation",
            ],
        },
        "device": {
            "hmm_requested": args.hmm_device,
            "hmm_runtime": args.hmm_runtime,
            "ht_search_xgb_requested": args.xgb_device,
            "ht_search_xgb_resolved": args.xgb_device_resolved,
        },
        "users": {"train": train_users, "cal": cal_users, "test": test_users},
        "best_config": {"H": int(best_h), "T": int(best_t)},
        "best_model": {
            "name": "gaussian_hmm_3state",
            "calibration": cal_name,
            "threshold_f1_on_cal": float(threshold),
            "kind": "single",
            "blend_weights": None,
        },
        "model_selection_on_cal": cal_records,
        "state_profile": state_profile,
        "samples": {"train": int(len(df_tr)), "cal": int(len(df_ca)), "test": int(len(df_te))},
        "positive_rate": {
            "train": float(y_tr.mean()),
            "cal": float(y_ca.mean()),
            "test": float(y_te.mean()),
        },
        "metrics_test": metrics_test,
        "ci_test": {
            "AUC-PR": {"mean": ci_ap[0], "low": ci_ap[1], "high": ci_ap[2]},
            "AUC-ROC": {"mean": ci_roc[0], "low": ci_roc[1], "high": ci_roc[2]},
            "Brier": {"mean": ci_brier[0], "low": ci_brier[1], "high": ci_brier[2]},
        },
        "search_results": search_records,
        "feature_cols": FEATURE_COLS,
    }

    with open(output_dir / "driver_state_hmm_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Driver-state-only 3-state HMM lookahead impairment predictor")
    ap.add_argument("--data-path", default="data")
    ap.add_argument("--output-dir", default="evaluation/driver_state_hmm_lookahead")
    ap.add_argument("--h-values", default="4,6,8,10,12,15,18")
    ap.add_argument("--t-values", default="1,2,3,4,5")
    ap.add_argument("--t-parsimony-tol", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)

    ap.add_argument("--hmm-states", type=int, default=3, help="Number of HMM latent states.")
    ap.add_argument("--hmm-max-iter", type=int, default=45)
    ap.add_argument("--hmm-tol", type=float, default=1e-3)
    ap.add_argument("--hmm-min-covar", type=float, default=1e-3)
    ap.add_argument("--hmm-device", choices=["auto", "cpu", "cuda"], default="auto")

    # Used only by H/T train-only search to preserve comparability with existing pipeline.
    ap.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--xgb-n-jobs", type=int, default=-1)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    args.h_values = parse_int_list(args.h_values)
    args.t_values = parse_int_list(args.t_values)
    args.xgb_device_resolved = resolve_xgb_device(args.xgb_device)
    if args.xgb_device_resolved == "cuda" and int(args.xgb_n_jobs) <= 0:
        args.xgb_n_jobs = 1
    args.hmm_runtime = resolve_hmm_runtime(args.hmm_device)
    LOG.info(
        "runtime: hmm_requested=%s hmm_backend=%s hmm_resolved=%s | ht_search_xgb=%s/%s",
        args.hmm_device,
        args.hmm_runtime["backend"],
        args.hmm_runtime["resolved"],
        args.xgb_device,
        args.xgb_device_resolved,
    )

    result = run_pipeline(args)
    m = result["metrics_test"]
    LOG.info(
        "FINAL TEST HMM: AUC-PR=%.4f AUC-ROC=%.4f Brier=%.4f F1=%.4f | H=%s T=%s cal=%s",
        m["AUC-PR"],
        m["AUC-ROC"],
        m["Brier"],
        m["F1"],
        result["best_config"]["H"],
        result["best_config"]["T"],
        result["best_model"]["calibration"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
