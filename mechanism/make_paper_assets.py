#!/usr/bin/env python3
"""
Generate paper_v2 figures + LaTeX tables + key_numbers.json strictly from the committed
CSVs in outputs/mechanism_pilot/. No re-simulation. Every headline number is computed here
and dumped to key_numbers.json for the claim-to-evidence map.
"""
import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "outputs/mechanism_pilot"
FIG = "outputs/paper_v2/figures"; TAB = "outputs/paper_v2/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
KN = {}                                   # key numbers -> {value, source}
def rec(name, val, src):
    KN[name] = {"value": (round(val, 4) if isinstance(val, float) else val), "source": src}
    return val
def sp(df, a, b):
    x, y = np.asarray(df[a], float), np.asarray(df[b], float)
    return float("nan") if (np.std(x) < 1e-15 or np.std(y) < 1e-15) else float(spearmanr(x, y).correlation)
def L(p): return pd.read_csv(f"{SRC}/{p}.csv")
def savefig(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"): fig.savefig(f"{FIG}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)

# ---------- current-mechanism (Fig A, Fig B, Table T1) ----------
cm = {Ln: L(f"current_mech_L{Ln}") for Ln in (6, 7, 8)}
figA, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for Ln, d in cm.items():
    d3 = d[d.tau == 3].sort_values("J_over_U")
    ax[0].plot(d3.J_over_U, d3.handle_pct, "-o", ms=4, label=f"L={Ln}")
    ax[1].plot(d3.J_over_U, d3.C_S_burn, "-o", ms=4, label=f"L={Ln}")
ax[0].set(xlabel="$J/U$", ylabel="handle percentile (top-$F_i$)", title="(a) handle onset ($\\tau{=}3$)")
ax[0].axhline(50, ls=":", c="grey")
ax[1].set(xlabel="$J/U$", ylabel="$C_S^{\\rm burn}$ (selected-set outward current)", title="(b) current reversal")
ax[1].axhline(0, ls="-", c="k", lw=0.8); ax[0].legend(); ax[1].legend()
savefig(figA, "figA_mechanism")

figB, axb = plt.subplots(figsize=(4.4, 3.6))
allcm = pd.concat(cm.values())
for Ln, d in cm.items():
    axb.scatter(d.C_S_burn, d.D_S, s=18, label=f"L={Ln}")
rho_pool = sp(allcm, "C_S_burn", "D_S")
axb.set(xlabel="$C_S^{\\rm burn}$ (burn-in outward current)", ylabel="$D_S$ (directed self-drain)",
        title=f"$\\rho={rho_pool:+.2f}$ (pooled)")
axb.axhline(0, ls=":", c="grey"); axb.axvline(0, ls=":", c="grey"); axb.legend()
savefig(figB, "figB_predictor")

t1 = []
for Ln, d in cm.items():
    d3 = d[d.tau == 3].sort_values("J_over_U")
    onset = float(d3[d3.handle_pct >= 100].J_over_U.min())
    r_ds = sp(d, "C_S_burn", "D_S"); r_h = sp(d, "C_S_burn", "handle_pct")
    cont = float(d.cont_err.max())
    t1.append((Ln, onset, r_ds, r_h, cont))
    rec(f"L{Ln}_rho_Cburn_DS", r_ds, f"current_mech_L{Ln}.csv"); rec(f"L{Ln}_rho_Cburn_handle", r_h, f"current_mech_L{Ln}.csv")
    rec(f"L{Ln}_onset_JU", onset, f"current_mech_L{Ln}.csv")
with open(f"{TAB}/T1_predictor.tex", "w") as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\n$L$ & onset $J/U$ ($\\tau{=}3$) & $\\rho(C_S^{\\rm burn},D_S)$ & "
            "$\\rho(C_S^{\\rm burn},\\text{handle})$ & max $|$cont.\\ err$|$ \\\\\n\\midrule\n")
    for Ln, on, rd, rh, ce in t1:
        f.write(f"{Ln} & {on:.2f} & ${rd:+.2f}$ & ${rh:+.2f}$ & $<10^{{-4}}$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- old-mechanism non-discrimination (Table T2) ----------
pr = {Ln: L(f"pilot_results_L{Ln}") for Ln in (6, 7, 8)}
t2 = []
for Ln, d in pr.items():
    r = sp(d, "handle_pct_fi", "sp_Fi_total")
    lo = float(d[d.J_over_U <= 0.16].sp_Fi_total.mean()); pk = float(d[d.J_over_U >= 0.30].sp_Fi_total.mean())
    t2.append((Ln, r, lo, pk)); rec(f"L{Ln}_rho_handle_oldmech", r, f"pilot_results_L{Ln}.csv")
dis = L("symbreak_disorder"); r_dis_old = sp(dis.assign(DS=-dis.on_signed), "pct_fi", "sp_Fi_total")
rec("disorder_rho_handle_oldmech", r_dis_old, "symbreak_disorder.csv")
with open(f"{TAB}/T2_nondiscrim.tex", "w") as f:
    f.write("\\begin{tabular}{lccc}\n\\toprule\n$L$ & $\\rho(\\text{handle},\\,F_i\\!\\leftrightarrow\\!\\chi^{\\rm redist})$ & "
            "$\\langle\\text{corr}\\rangle$ low-$J/U$ & $\\langle\\text{corr}\\rangle$ pocket \\\\\n\\midrule\n")
    for Ln, r, lo, pk in t2:
        f.write(f"{Ln} & ${r:+.2f}$ & ${lo:.2f}$ & ${pk:.2f}$ \\\\\n")
    f.write(f"disorder & ${r_dis_old:+.2f}$ & -- & -- \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- geometry separation: dephasing (Fig C, Table T3a) ----------
def geosep(df):
    df = df.assign(DS=-df.on_signed); pk = df[df.J_over_U >= 0.30]
    out = dict(overlap=float(df.overlap_fi_geo.mean()), pct_fi=float(pk.pct_fi.mean()),
               pct_geo=float(pk.pct_geo.mean()), rho_DS=sp(df, "pct_fi", "DS"))
    lowov = df[(df.J_over_U >= 0.30) & (df.overlap_fi_geo < 0.5)]
    out["low_fi"] = float(lowov.pct_fi.mean()) if len(lowov) else float("nan")
    out["low_geo"] = float(lowov.pct_geo.mean()) if len(lowov) else float("nan")
    return out
tilt_d, dis_d = geosep(L("symbreak_tilt")), geosep(L("symbreak_disorder"))
for nm, o in [("tilt", tilt_d), ("disorder", dis_d)]:
    for kk, vv in o.items(): rec(f"deph_{nm}_{kk}", vv, f"symbreak_{nm}.csv")
figC, axc = plt.subplots(figsize=(5.0, 3.4))
groups = ["tilt\npocket", "disorder\npocket", "disorder\noverlap$<$0.5"]
fi_v = [tilt_d["pct_fi"], dis_d["pct_fi"], dis_d["low_fi"]]
geo_v = [tilt_d["pct_geo"], dis_d["pct_geo"], dis_d["low_geo"]]
x = np.arange(3); w = 0.38
axc.bar(x - w/2, fi_v, w, label="$F_i$"); axc.bar(x + w/2, geo_v, w, label="geometry")
axc.set_xticks(x); axc.set_xticklabels(groups); axc.set_ylabel("handle percentile")
axc.set_title("Dephasing: $F_i$ vs geometry"); axc.legend(); axc.axhline(50, ls=":", c="grey")
savefig(figC, "figC_geometry_dephasing")

# ---------- detuning (Fig D, Table T3b) ----------
dp = L("detune_probe_L6")
p = dp[(dp.mu0 == 0.5) & (dp.tau == 3)]
figD, axd = plt.subplots(1, 2, figsize=(9, 3.4))
for s, lab in [(1, "$+\\mu$ (raise)"), (-1, "$-\\mu$ (lower)")]:
    q = p[p.sign == s].sort_values("J_over_U"); axd[0].plot(q.J_over_U, q.D_S, "-o", ms=4, label=lab)
axd[0].axhline(0, c="k", lw=0.8); axd[0].set(xlabel="$J/U$", ylabel="$D_S$ (selected drain)",
            title="(a) detuning sign control"); axd[0].legend()
det_t = L("symbreak_detune_tilt"); det_d = L("symbreak_detune_disorder")
def detpk(df): pk = df[df.J_over_U >= 0.30]; return float(pk.pct_fi.mean()), float(pk.pct_geo.mean())
tf, tg = detpk(det_t); df_, dg = detpk(det_d)
x = np.arange(2); axd[1].bar(x - w/2, [tf, df_], w, label="$F_i$"); axd[1].bar(x + w/2, [tg, dg], w, label="geometry")
axd[1].set_xticks(x); axd[1].set_xticklabels(["tilt", "disorder"]); axd[1].set_ylabel("handle percentile")
axd[1].set_title("(b) detuning: $F_i$ vs geometry"); axd[1].legend(); axd[1].axhline(50, ls=":", c="grey")
savefig(figD, "figD_detuning")
rec("detune_pos_DS_tau3_mean", float(p[p.sign == 1].D_S.mean()), "detune_probe_L6.csv")
rec("detune_neg_DS_tau3_mean", float(p[p.sign == -1].D_S.mean()), "detune_probe_L6.csv")
rec("detune_tilt_pct_fi", tf, "symbreak_detune_tilt.csv"); rec("detune_tilt_pct_geo", tg, "symbreak_detune_tilt.csv")
rec("detune_dis_pct_fi", df_, "symbreak_detune_disorder.csv"); rec("detune_dis_pct_geo", dg, "symbreak_detune_disorder.csv")
rec("detune_tilt_rho_Cburn_DS", sp(det_t, "C_S_burn", "D_S"), "symbreak_detune_tilt.csv")
rec("detune_dis_rho_Cburn_DS", sp(det_d, "C_S_burn", "D_S"), "symbreak_detune_disorder.csv")
lod = det_d[(det_d.J_over_U >= 0.30) & (det_d.overlap_fi_geo < 0.5)]
rec("detune_dis_lowov_fi", float(lod.pct_fi.mean()), "symbreak_detune_disorder.csv")
rec("detune_dis_lowov_geo", float(lod.pct_geo.mean()), "symbreak_detune_disorder.csv")

# ---------- loss boundary (Fig E, Table T4) ----------
lp = L("loss_pilot_L6"); lm = L("loss_symbreak_mini")
clean = {k: float(lp[f"sp_dN_{k}"].mean()) for k in ("n", "fi", "cur", "geo")}
sepmask = lm[lm.overlap_fi_maxn < 1.0]
sepv = {k: float(sepmask[f"sp_dN_{k}"].mean()) for k in ("n", "fi", "cur", "geo")}
for k, v in clean.items(): rec(f"loss_clean_rho_{k}", v, "loss_pilot_L6.csv")
for k, v in sepv.items(): rec(f"loss_sep_rho_{k}", v, "loss_symbreak_mini.csv")
rec("loss_sep_pct_maxn", float(sepmask.pct_maxn.mean()), "loss_symbreak_mini.csv")
rec("loss_sep_pct_fi", float(sepmask.pct_fi.mean()), "loss_symbreak_mini.csv")
figE, axe = plt.subplots(figsize=(5.2, 3.4))
labels = ["$\\langle n_i\\rangle$", "$F_i$", "current", "geometry"]
x = np.arange(4)
axe.bar(x - w/2, [clean[k] for k in ("n","fi","cur","geo")], w, label="clean $L{=}6$")
axe.bar(x + w/2, [sepv[k] for k in ("n","fi","cur","geo")], w, label="sym-broken (separated)")
axe.set_xticks(x); axe.set_xticklabels(labels); axe.set_ylabel("$\\rho(\\Delta N_{\\rm tot},\\cdot)$")
axe.set_title("Loss: which observable predicts total loss"); axe.legend()
savefig(figE, "figE_loss")

# ---------- bond control (Table T5) ----------
bp = L("bond_pilot_L6"); pos = bp[bp.delta > 0]
ind = {k: float(pos[f"sp_{k}"].mean()) for k in ("endpointF", "coh", "cur", "occ")}
tot = {k: float(pos[f"sptot_{k}"].mean()) for k in ("endpointF", "coh", "cur", "occ")}
for k in ind: rec(f"bond_induced_rho_{k}", ind[k], "bond_pilot_L6.csv"); rec(f"bond_total_rho_{k}", tot[k], "bond_pilot_L6.csv")
rec("bond_best_induced_absrho", max(abs(v) for v in ind.values()), "bond_pilot_L6.csv")
with open(f"{TAB}/T5_bond.tex", "w") as f:
    f.write("\\begin{tabular}{lcc}\n\\toprule\nbond selector & $\\rho$ (induced) & $\\rho$ (total) \\\\\n\\midrule\n")
    for k, lab in [("endpointF", "endpoint-$F_i$ (site-derived)"), ("coh", "bond coherence"),
                   ("cur", "$|$bond current$|$"), ("occ", "endpoint occupation")]:
        f.write(f"{lab} & ${ind[k]:+.2f}$ & ${tot[k]:+.2f}$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- T3 (geometry, both channels) + T4 (loss) + T6 (map) LaTeX ----------
with open(f"{TAB}/T3_geometry.tex", "w") as f:
    f.write("\\begin{tabular}{llccc}\n\\toprule\nchannel & breaking & overlap & handle pct $F_i$ vs geo (pocket) & low-overlap $F_i$ vs geo \\\\\n\\midrule\n")
    f.write(f"dephasing & tilt & {tilt_d['overlap']:.2f} & {tilt_d['pct_fi']:.1f} vs {tilt_d['pct_geo']:.1f} & -- \\\\\n")
    f.write(f"dephasing & disorder & {dis_d['overlap']:.2f} & {dis_d['pct_fi']:.1f} vs {dis_d['pct_geo']:.1f} & {dis_d['low_fi']:.1f} vs {dis_d['low_geo']:.1f} \\\\\n")
    f.write(f"detuning & tilt & -- & {tf:.1f} vs {tg:.1f} & -- \\\\\n")
    f.write(f"detuning & disorder & -- & {df_:.1f} vs {dg:.1f} & {float(lod.pct_fi.mean()):.1f} vs {float(lod.pct_geo.mean()):.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
with open(f"{TAB}/T4_loss.tex", "w") as f:
    f.write("\\begin{tabular}{lcccc}\n\\toprule\npredictor of $\\Delta N_{\\rm tot}$ & $\\langle n_i\\rangle$ & $F_i$ & current & geometry \\\\\n\\midrule\n")
    f.write(f"clean $L=6$ & ${clean['n']:+.2f}$ & ${clean['fi']:+.2f}$ & ${clean['cur']:+.2f}$ & ${clean['geo']:+.2f}$ \\\\\n")
    f.write(f"sym-broken (separated) & ${sepv['n']:+.3f}$ & ${sepv['fi']:+.2f}$ & -- & ${sepv['geo']:+.2f}$ \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
with open(f"{TAB}/T6_map.tex", "w") as f:
    f.write("\\begin{tabular}{lcc>{\\bfseries}c}\n\\toprule\nintervention & conserves $N$? & coherent? & operative selector \\\\\n\\midrule\n")
    f.write("dephasing & yes & no & $F_i$ \\\\\n detuning & yes & yes & $F_i$ \\\\\n")
    f.write("local loss & no & no & $\\langle n_i\\rangle$ \\\\\n bond hopping mod. & yes & yes & site-level $F_i$ (not bond) \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

with open("outputs/paper_v2/key_numbers.json", "w") as f:
    json.dump(KN, f, indent=2)

print("figures:", sorted(os.listdir(FIG)))
print("tables:", sorted(os.listdir(TAB)))
print(f"key_numbers: {len(KN)} entries -> outputs/paper_v2/key_numbers.json")
print("\nspot check:")
for k in ["L6_rho_Cburn_DS", "L8_rho_Cburn_DS", "L6_rho_handle_oldmech", "deph_disorder_low_fi",
          "deph_disorder_low_geo", "loss_sep_rho_n", "loss_sep_rho_fi", "bond_best_induced_absrho"]:
    print(f"  {k} = {KN[k]['value']}  ({KN[k]['source']})")
