import itertools
import numpy as np
from scipy.stats import spearmanr
import bh, bh_hardening as bhh


def bond_kinetic_op(b, basis, idx_map, nmax):
    D = len(basis); K = np.zeros((D, D))
    for s, state in enumerate(basis):
        ni, nj = state[b], state[b + 1]
        if nj > 0 and ni < nmax:
            new = list(state); new[b] = ni + 1; new[b + 1] = nj - 1
            j = idx_map.get(tuple(new))
            if j is not None: K[j, s] += np.sqrt((ni + 1) * nj)
        if ni > 0 and nj < nmax:
            new = list(state); new[b] = ni - 1; new[b + 1] = nj + 1
            j = idx_map.get(tuple(new))
            if j is not None: K[j, s] += np.sqrt(ni * (nj + 1))
    return K


def compute(L, ju, tau, threshold):
    bh._SPARSE_D_THRESHOLD = threshold
    cond = bhh.build_condition(L, ju)
    Fi = cond["Fi"]; occ_burn = cond["occ_burn"]; rho = cond["rho_burn"]
    basis = cond["basis"]; idx = cond["idx_map"]; k = bhh.k_sites(L)
    S = tuple(sorted(int(s) for s in np.argsort(Fi)[-k:])); Sset = set(S)
    losses = {}
    for sub in itertools.combinations(range(L), k):
        occ_after = bhh.occ_from_rho(bhh.evolve_with_extra(cond, list(sub), tau, bhh.GAMMA_EXTRA), cond)
        losses[sub] = sum(max(0.0, occ_burn[i] - occ_after[i]) for i in sub)
    allv = np.array(list(losses.values())); pct = 100.0 * np.mean(allv <= losses[S])
    occ_base_t = bhh.occ_from_rho(bh.evolve_rho(rho, cond["liouv_base"], tau), cond)
    R = np.zeros((L, L))
    for i in range(L):
        occ_pert = bhh.occ_from_rho(bhh.evolve_with_extra(cond, [i], tau, 0.1), cond)
        R[:, i] = (occ_base_t - occ_pert) / 0.1
    total = np.abs(R).sum(axis=0)
    occ_after_S = bhh.occ_from_rho(bhh.evolve_with_extra(cond, list(S), tau, bhh.GAMMA_EXTRA), cond)
    dn = occ_after_S - occ_burn; tot = float(np.abs(dn).sum())
    off = float(sum(abs(dn[j]) for j in range(L) if j not in Sset) / tot)
    on = float(sum(dn[i] for i in S))
    Cb = np.array([np.real(np.trace(bond_kinetic_op(b, basis, idx, bhh.NMAX) @ rho)) for b in range(L - 1)])
    return dict(Fi=Fi, occ=occ_burn, S=S, pct=pct, R=R, off=off, on=on, Cb=Cb,
                spft=float(spearmanr(Fi, total).correlation))


L, ju, tau = 7, 0.30, 3
print(f"PARITY CHECK  L={L}  J/U={ju}  tau={tau}   (dense thr=100 vs forced-sparse thr=60)")
d = compute(L, ju, tau, 100)
s = compute(L, ju, tau, 60)
md = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
print(f"  selected S      : dense={d['S']} sparse={s['S']}  match={d['S']==s['S']}")
print(f"  max|dFi|        : {md(d['Fi'],s['Fi']):.2e}")
print(f"  max|docc_burn|  : {md(d['occ'],s['occ']):.2e}")
print(f"  max|dR_kernel|  : {md(d['R'],s['R']):.2e}")
print(f"  max|dCb|        : {md(d['Cb'],s['Cb']):.2e}")
print(f"  handle pct      : dense={d['pct']:.3f} sparse={s['pct']:.3f}  |d|={abs(d['pct']-s['pct']):.2e}")
print(f"  off_sel_frac    : dense={d['off']:.6f} sparse={s['off']:.6f}  |d|={abs(d['off']-s['off']):.2e}")
print(f"  on_sel_signed   : dense={d['on']:.6f} sparse={s['on']:.6f}  |d|={abs(d['on']-s['on']):.2e}")
print(f"  Spearman(Fi,tot): dense={d['spft']:.4f} sparse={s['spft']:.4f}")
tol = 1e-8
ok = (d["S"] == s["S"] and md(d["Fi"], s["Fi"]) < tol and md(d["occ"], s["occ"]) < tol
      and md(d["R"], s["R"]) < tol and abs(d["pct"] - s["pct"]) < tol
      and abs(d["off"] - s["off"]) < tol and abs(d["on"] - s["on"]) < tol)
print(f"\nPARITY {'PASS' if ok else 'FAIL'}  (tol={tol})")
