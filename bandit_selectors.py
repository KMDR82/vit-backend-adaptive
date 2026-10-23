# Non-stationary bandit selectors with switching costs.
# Notation follows Sect. 3 of the manuscript:
#   cost c_t = L + lambda_E*E + lambda_D*1[deadline miss]   (Eq. 1)
#   C_T(pi) = sum c_t + sum S_{a->a'} 1[switch]             (Eq. 2)
#   switching-aware value adds S_{a_prev->a}/H              (Eq. 4)
# All selectors MINIMIZE cost. Self-test: python selectors.py
import numpy as np

class BaseSelector:
    """K arms; observe(cost) after each select(). Switch penalty S/H charged
    inside the decision rule when switch_costs given (Eq. 4 variant)."""
    def __init__(self, K, switch_costs=None, H=100, rng=None):
        self.K = K
        self.S = np.zeros((K, K)) if switch_costs is None else np.asarray(switch_costs, float)
        self.H = H
        self.rng = rng or np.random.default_rng(0)
        self.prev = None
        self.t = 0
    def _switch_term(self):
        if self.prev is None: return np.zeros(self.K)
        pen = self.S[self.prev].copy() / self.H
        pen[self.prev] = 0.0
        return pen
    def select(self): raise NotImplementedError
    def observe(self, arm, cost): raise NotImplementedError

class RandomSelector(BaseSelector):
    def select(self):
        self.t += 1; a = int(self.rng.integers(self.K)); self.prev = a; return a
    def observe(self, arm, cost): pass

class EpsGreedy(BaseSelector):
    def __init__(self, K, eps=0.1, window=200, **kw):
        super().__init__(K, **kw); self.eps = eps; self.window = window
        self.hist = [[] for _ in range(K)]
    def select(self):
        self.t += 1
        if self.prev is None or self.rng.random() < self.eps or any(len(h) == 0 for h in self.hist):
            a = int(self.rng.integers(self.K))
        else:
            means = np.array([np.mean(h[-self.window:]) for h in self.hist])
            a = int(np.argmin(means + self._switch_term()))
        self.prev = a; return a
    def observe(self, arm, cost): self.hist[arm].append(cost)

class SWUCB(BaseSelector):
    """Sliding-window UCB (Garivier & Moulines 2011), cost-minimizing form; Eq. 4."""
    def __init__(self, K, tau=300, xi=1.0, **kw):
        super().__init__(K, **kw); self.tau = tau; self.xi = xi
        self.arms, self.costs = [], []
    def select(self):
        self.t += 1
        arms = np.array(self.arms[-self.tau:]); costs = np.array(self.costs[-self.tau:])
        n = np.array([(arms == a).sum() for a in range(self.K)])
        if (n == 0).any():
            a = int(np.argmin(n))
        else:
            mu = np.array([costs[arms == a].mean() for a in range(self.K)])
            bonus = self.xi * np.sqrt(np.log(min(self.t, self.tau)) / n)
            a = int(np.argmin(mu - bonus + self._switch_term()))
        self.prev = a; return a
    def observe(self, arm, cost): self.arms.append(arm); self.costs.append(cost)

class DUCB(BaseSelector):
    """Discounted UCB: geometric forgetting gamma."""
    def __init__(self, K, gamma=0.99, xi=1.0, **kw):
        super().__init__(K, **kw); self.g = gamma; self.xi = xi
        self.N = np.zeros(K); self.Csum = np.zeros(K)
    def select(self):
        self.t += 1
        if (self.N < 1e-9).any():
            a = int(np.argmin(self.N))
        else:
            mu = self.Csum / self.N
            bonus = self.xi * np.sqrt(np.log(self.N.sum() + 1) / self.N)
            a = int(np.argmin(mu - bonus + self._switch_term()))
        self.prev = a; return a
    def observe(self, arm, cost):
        self.N *= self.g; self.Csum *= self.g
        self.N[arm] += 1; self.Csum[arm] += cost

class EXP3S(BaseSelector):
    """EXP3.S (Auer et al. 2002) for rewards in [0,1]; costs mapped via
    reward = 1 - clip(cost/cost_scale). cost_scale set from data."""
    def __init__(self, K, eta=0.05, alpha=0.001, cost_scale=1.0, **kw):
        super().__init__(K, **kw); self.eta = eta; self.alpha = alpha
        self.scale = cost_scale; self.w = np.ones(K)
    def _probs(self):
        p = self.w / self.w.sum()
        return (1 - self.eta) * p + self.eta / self.K
    def select(self):
        self.t += 1
        p = self._probs()
        # switching-aware: tilt probabilities away from expensive switches
        pen = np.exp(-self._switch_term() / max(self.scale, 1e-9))
        q = p * pen; q /= q.sum()
        a = int(self.rng.choice(self.K, p=q)); self.prev = a; self._lastp = q
        return a
    def observe(self, arm, cost):
        r = 1.0 - min(max(cost / self.scale, 0.0), 1.0)
        xhat = r / self._lastp[arm]
        self.w[arm] *= np.exp(self.eta * xhat / self.K)
        self.w += (np.e * self.alpha / self.K) * self.w.sum()   # egalitarian mixing
        self.w /= self.w.max()                                   # numeric guard

class GLRklUCB(BaseSelector):
    """Simplified GLR change-point restart around a UCB core (Besson et al. 2022):
    per-arm Gaussian GLR on windowed means; restart that arm's stats on detection."""
    def __init__(self, K, xi=1.0, glr_thresh=6.0, min_len=30, **kw):
        super().__init__(K, **kw); self.xi = xi; self.th = glr_thresh; self.ml = min_len
        self.hist = [[] for _ in range(K)]; self.restarts = np.zeros(K, int)
    def _glr_detect(self, h):
        n = len(h)
        if n < 2 * self.ml: return False
        x = np.array(h); s = x.std() + 1e-9
        best = 0.0
        for k in range(self.ml, n - self.ml):
            m1, m2 = x[:k].mean(), x[k:].mean()
            stat = k * (n - k) / n * ((m1 - m2) / s) ** 2 / 2
            best = max(best, stat)
        return best > self.th
    def select(self):
        self.t += 1
        n = np.array([len(h) for h in self.hist])
        if (n == 0).any():
            a = int(np.argmin(n))
        else:
            mu = np.array([np.mean(h) for h in self.hist])
            bonus = self.xi * np.sqrt(np.log(self.t) / n)
            a = int(np.argmin(mu - bonus + self._switch_term()))
        self.prev = a; return a
    def observe(self, arm, cost):
        self.hist[arm].append(cost)
        if len(self.hist[arm]) % 20 == 0 and self._glr_detect(self.hist[arm]):
            self.hist[arm] = self.hist[arm][-self.ml:]
            self.restarts[arm] += 1

def replay(env_costs, selector, switch_costs):
    """env_costs: (T,K) realized cost of each arm at each t (from traces).
    Returns dict with cumulative cost incl. switch charges (Eq. 2)."""
    T, K = env_costs.shape
    total, switches, choices = 0.0, 0, []
    prev = None
    for t in range(T):
        a = selector.select()
        c = float(env_costs[t, a])
        selector.observe(a, c)
        total += c
        if prev is not None and a != prev:
            total += float(switch_costs[prev, a]); switches += 1
        prev = a; choices.append(a)
    return {"C_T": total, "switches": switches, "choices": np.array(choices)}

def baselines(env_costs, switch_costs, phase_ids):
    """best-static a*, and per-phase oracle (pays switch costs at breakpoints)."""
    T, K = env_costs.shape
    static_tot = env_costs.sum(axis=0)
    a_star = int(np.argmin(static_tot))
    oracle_tot, prev = 0.0, None
    for ph in np.unique(phase_ids):
        m = phase_ids == ph
        a = int(np.argmin(env_costs[m].sum(axis=0)))
        oracle_tot += env_costs[m][:, a].sum()
        if prev is not None and a != prev:
            oracle_tot += float(switch_costs[prev, a])
        prev = a
    return {"a_star": a_star, "C_static": float(static_tot[a_star]),
            "C_oracle": float(oracle_tot)}

if __name__ == "__main__":
    # --- synthetic self-test: 3 arms, 4 phases, best arm changes each phase ---
    rng = np.random.default_rng(42)
    K, phase_len, phases = 3, 500, [0, 1, 2, 0]
    T = phase_len * len(phases)
    means = np.array([[2.0, 3.0, 4.0], [4.0, 2.0, 3.0], [3.0, 4.0, 2.0]])
    env = np.zeros((T, K)); phase_ids = np.zeros(T, int)
    for i, best in enumerate(phases):
        sl = slice(i * phase_len, (i + 1) * phase_len)
        env[sl] = rng.normal(means[best], 0.3, size=(phase_len, K))
        phase_ids[sl] = i
    S = np.full((K, K), 15.0); np.fill_diagonal(S, 0.0)

    base = baselines(env, S, phase_ids)
    print(f"best-static arm={base['a_star']}  C_static={base['C_static']:.0f}  "
          f"C_oracle={base['C_oracle']:.0f}")
    mk = {
        "random":   lambda: RandomSelector(K, switch_costs=S, rng=np.random.default_rng(1)),
        "eps":      lambda: EpsGreedy(K, switch_costs=S, rng=np.random.default_rng(1)),
        "SW-UCB":   lambda: SWUCB(K, tau=300, switch_costs=S),
        "D-UCB":    lambda: DUCB(K, gamma=0.995, switch_costs=S),
        "EXP3.S":   lambda: EXP3S(K, cost_scale=6.0, switch_costs=S, rng=np.random.default_rng(1)),
        "GLR-klUCB":lambda: GLRklUCB(K, switch_costs=S),
    }
    print(f"{'selector':10s} {'C_T':>8s} {'regret':>8s} {'delta_vs_static':>16s} {'switches':>9s}")
    for name, f in mk.items():
        r = replay(env, f(), S)
        print(f"{name:10s} {r['C_T']:8.0f} {r['C_T']-base['C_oracle']:8.0f} "
              f"{base['C_static']-r['C_T']:16.0f} {r['switches']:9d}")
