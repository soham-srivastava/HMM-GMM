"""
The gate: a Gaussian HMM over a small set of market observables.

Three things this wrapper fixes over calling hmmlearn directly:

1. FILTERING.  To forecast the state at t we need
       P(S_{t-1} | r_1..r_{t-1})  then  P(S_t) = P(S_{t-1}) @ A
   so the model must see the whole history, not one observation.  hmmlearn's
   predict_proba returns the smoothed posterior P(S_t | O_1..O_T); at the final
   index there is no future to condition on, so its last row IS the filtered
   belief and uses nothing past t-1.

2. LABEL SWITCHING.  The likelihood is invariant to renaming states, so refits
   permute them.  States are sorted by fitted mean return, making state 0 the
   lowest-drift state at every refit.

3. SCALE / LOCAL OPTIMA.  Observables are standardised inside the fitting
   window and EM is restarted from several seeds (best likelihood wins).
   Without this a multi-feature HMM regularly collapses to one occupied state.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from hmmlearn import hmm

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

N_ITER, TOL, N_RESTARTS = 100, 1e-4, 5


class RegimeModel:
    def __init__(self, n_states=3, seed=100):
        self.n_states = n_states
        self.seed = seed
        self.model = None
        self.perm = None
        self.converged = False

    # ---------------- fitting ----------------
    def _prep(self, X, fit=False):
        obs = np.asarray(X, dtype=float).reshape(len(X), -1)
        if fit:
            self._mu = obs.mean(0)
            self._sd = np.where(obs.std(0) < 1e-12, 1.0, obs.std(0))
        return (obs - self._mu) / self._sd

    def fit(self, X):
        obs = self._prep(X, fit=True)
        best, best_ll = None, -np.inf
        for k in range(N_RESTARTS):
            m = hmm.GaussianHMM(n_components=self.n_states, covariance_type="diag",
                                n_iter=N_ITER, tol=TOL, random_state=self.seed + k)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(obs)
                    ll = float(m.score(obs))
            except Exception:
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError("all HMM restarts failed")
        self.model, self.loglik = best, best_ll
        self.converged = bool(getattr(best.monitor_, "converged", False))
        self.perm = np.argsort(np.asarray(best.means_)[:, 0])   # ascending mean
        return self

    # ---------------- parameters, in original units ----------------
    @staticmethod
    def _diag(m):
        c = np.asarray(m.covars_)
        return np.array([np.diag(x) for x in c]) if c.ndim == 3 else np.atleast_2d(c)

    @property
    def transmat(self):
        return np.asarray(self.model.transmat_)[np.ix_(self.perm, self.perm)]

    @property
    def means(self):
        return np.asarray(self.model.means_)[self.perm] * self._sd + self._mu

    @property
    def variances(self):
        return self._diag(self.model)[self.perm] * self._sd ** 2

    # ---------------- inference ----------------
    def posteriors(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict_proba(self._prep(X))[:, self.perm]

    def next_state_probs(self, X):
        """P(S_{T+1} | O_1..O_T). Pass the whole history, never one row."""
        return self.posteriors(X)[-1] @ self.transmat

    def hard_states(self, X):
        return self.posteriors(X).argmax(1)

    # ---------------- structure ----------------
    def stationary_distribution(self):
        vals, vecs = np.linalg.eig(self.transmat.T)
        v = np.abs(np.real(vecs[:, np.argmin(np.abs(vals - 1.0))]))
        return v / v.sum()

    def expected_durations(self):
        """1/(1 - A_ii): average number of bars a state lasts."""
        return 1.0 / np.clip(1.0 - np.diag(self.transmat), 1e-6, None)

    def persistence(self):
        """
        lambda2, the second-largest eigenvalue of A, is the decay rate of state
        information: P(S_{t+k}|S_t) - pi ~ lambda2^k.  Near 1 means real,
        persistent regimes.  Near 0 means A is already the stationary
        distribution, the gate carries no forward information, and the model is
        effectively a fixed-weight ensemble of the experts.
        """
        lam = np.sort(np.abs(np.linalg.eigvals(self.transmat)))
        l2 = float(lam[-2])
        return {"lambda2": l2,
                "half_life_bars": float(np.log(0.5) / np.log(l2)) if 0 < l2 < 1 else np.inf}

    def summary(self, periods_per_year=365):
        """
        What each state is, in economic terms, with a label READ OFF the fitted
        numbers -- not assigned in advance.

        drift_t = mu * sqrt(d) / sigma is the t-statistic of the drift over one
        typical episode of length d: it asks whether a state's return is
        distinguishable from zero at the horizon the state actually lasts.
        A state with |drift_t| < 0.5 is a no-drift state; if it also carries the
        highest volatility, that is what "choppy" means.
        """
        mu, sd = self.means[:, 0], np.sqrt(np.abs(self.variances[:, 0]))
        dur = self.expected_durations()
        drift_t = mu * np.sqrt(dur) / np.where(sd > 0, sd, np.nan)
        vol_rank = np.argsort(np.argsort(sd))
        n = self.n_states

        labels = []
        for i in range(n):
            d = "bull" if drift_t[i] > 0.5 else "bear" if drift_t[i] < -0.5 else "flat"
            v = ("high-vol" if vol_rank[i] == n - 1 else
                 "low-vol" if vol_rank[i] == 0 else "mid-vol")
            labels.append(f"{d}/{v}")

        return pd.DataFrame({
            "mean_ret_daily_%": mu * 100,
            "vol_daily_%": sd * 100,
            "ann_return_%": ((1 + mu) ** periods_per_year - 1) * 100,
            "ann_vol_%": sd * np.sqrt(periods_per_year) * 100,
            "drift_t_per_episode": drift_t,
            "self_transition": np.diag(self.transmat),
            "expected_duration_bars": dur,
            "stationary_prob": self.stationary_distribution(),
            "label": labels,
        }, index=[f"state_{i}" for i in range(n)])


def fit_regime_model(X, n_states=3, seed=100):
    """Returns None if the model cannot be fitted at all."""
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < max(50, 20 * n_states):
        return None
    try:
        return RegimeModel(n_states, seed).fit(X)
    except Exception:
        return None
