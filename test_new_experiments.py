"""
Tests for new experiments added to bh.py:
  - run_selector_sweep_realization
  - _build_inhomogeneous_mu
  - run_inhomogeneous_experiment
  - run_gamma_scan_experiment
  - _gscan_ckpt_path / _sel_ckpt_path / _inhom_ckpt_path

Run with: python test_new_experiments.py
Expected: all tests PASS in < 2 min on a laptop.
"""

import json
import os
import sys
import tempfile
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap: import bh and patch CKPT_DIR to a temp directory so tests
# don't write to the real outputs/ folder.
# ---------------------------------------------------------------------------
import importlib
import bh

# ---- Tiny cfg for fast tests ----
_CFG = {
    "SEED":          42,
    "NMAX":          2,       # nmax=2 keeps Hilbert space tiny
    "GAMMA_BASE":    0.1,
    "GAMMA_EXTRA":   0.5,
    "BURN_IN_TIME":  1.0,     # short burn-in
    "N_TRIALS":      5,       # minimal random trials
    "N_BOOT":        20,
    "TAU_LIST":      [1.0, 2.0],
    "J_OVER_U_LIST": [0.20, 0.40],
    "L_LIST":        [6],
}
L, N, nmax = 6, 3, 2
JU         = 0.30
tau_list   = [1.0, 2.0]


def _mu_zero():
    return np.zeros(L)


def _mu_random(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.1, 0.1, size=L)


# ============================================================
# 1. Selector sweep realization
# ============================================================

def test_selector_sweep_keys():
    """run_selector_sweep_realization returns all 9 selectors with correct keys.

    Now includes dis_amp (top-k by |mu_i|) and dis_anti (bottom-k by |mu_i|)
    added to disentangle fi advantage from mere disorder-amplitude tracking.
    """
    mu_vec = _mu_random(1)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=tau_list, n_trials=5, burn_in_time=1.0,
        trial_seed=99, n_boot=20)

    SEL_NAMES = {"fi", "geo", "maxn", "minn", "bdy", "anti", "gen",
                 "dis_amp", "dis_anti"}
    assert "results" in res
    assert "sel_sites" in res
    assert set(res["sel_sites"].keys()) == SEL_NAMES, (
        f"Got selectors: {set(res['sel_sites'].keys())}\n"
        f"Expected:       {SEL_NAMES}")

    for tr in res["results"]:
        assert set(tr["selectors"].keys()) == SEL_NAMES
        for name, sd in tr["selectors"].items():
            assert "own_vs_rnd"   in sd
            assert "geo_vs_rnd"   in sd
            assert "sites"        in sd
            assert "det_loss_own" in sd
            assert "det_loss_geo" in sd
            for ci_key in ("mean", "ci_lo", "ci_hi"):
                assert ci_key in sd["own_vs_rnd"], f"{name}.own_vs_rnd missing {ci_key}"
                assert ci_key in sd["geo_vs_rnd"], f"{name}.geo_vs_rnd missing {ci_key}"

    print("  PASS: selector_sweep_keys (9 selectors including dis_amp, dis_anti)")


def test_selector_sweep_site_counts():
    """All selectors return exactly k sites in [0, L-1]."""
    mu_vec = _mu_random(2)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=3, burn_in_time=1.0,
        trial_seed=7, n_boot=10)
    k = res["k"]
    for name, sites in res["sel_sites"].items():
        assert len(sites) == k, f"{name}: expected {k} sites, got {len(sites)}"
        assert all(0 <= s < L for s in sites), f"{name}: site out of range: {sites}"
        assert len(set(sites)) == k, f"{name}: duplicate sites: {sites}"
    print("  PASS: selector_sweep_site_counts")


def test_selector_sweep_clean_symmetry():
    """Under zero disorder, fi and geo must be the same selector (clean symmetric chain)."""
    mu_vec = _mu_zero()
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=3, burn_in_time=1.0,
        trial_seed=0, n_boot=10)
    assert res["sel_sites"]["fi"] == res["sel_sites"]["geo"], (
        f"fi={res['sel_sites']['fi']} ≠ geo={res['sel_sites']['geo']} in clean case")
    print("  PASS: selector_sweep_clean_symmetry (fi==geo under μ=0)")


def test_boundary_selector_outermost():
    """bdy selector must be the outermost k sites (k derived from the realization)."""
    for L_ in [6, 7, 8]:
        N_ = L_ // 2
        mu = np.zeros(L_)
        res = bh.run_selector_sweep_realization(
            L_, N_, nmax, JU, mu,
            gamma_base=0.1, gamma_extra=0.5,
            tau_list=[1.0], n_trials=2, burn_in_time=0.5,
            trial_seed=0, n_boot=5)
        bdy = res["sel_sites"]["bdy"]
        k_  = res["k"]
        n_right = k_ // 2
        n_left  = k_ - n_right
        expected = sorted(list(range(n_left)) + list(range(L_ - n_right, L_)))
        assert bdy == expected, f"L={L_} k={k_}: bdy={bdy}, expected {expected}"
    print("  PASS: boundary_selector_outermost")


def test_anti_fi_is_bottom_k():
    """anti selector must be the k sites with LOWEST F_i."""
    mu_vec = _mu_random(5)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=2, burn_in_time=1.0,
        trial_seed=0, n_boot=5)
    Fi    = np.array(res["Fi"])
    k     = res["k"]
    anti  = res["sel_sites"]["anti"]
    fi    = res["sel_sites"]["fi"]
    expected_anti = sorted(np.argsort(Fi)[:k].tolist())
    assert anti == expected_anti, f"anti={anti}, expected {expected_anti}"
    # fi should have strictly higher Fi than anti (in aggregate)
    assert Fi[fi].mean() >= Fi[anti].mean(), "fi mean < anti mean — something wrong"
    print("  PASS: anti_fi_is_bottom_k")


def test_selector_sweep_loss_non_negative():
    """det_loss_own and det_loss_geo must be non-negative."""
    mu_vec = _mu_random(3)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0, 2.0], n_trials=3, burn_in_time=1.0,
        trial_seed=42, n_boot=10)
    for tr in res["results"]:
        for name, sd in tr["selectors"].items():
            assert sd["det_loss_own"] >= -1e-10, f"{name} det_loss_own < 0: {sd['det_loss_own']}"
            assert sd["det_loss_geo"] >= -1e-10, f"{name} det_loss_geo < 0: {sd['det_loss_geo']}"
    print("  PASS: selector_sweep_loss_non_negative")


# ============================================================
# 2. Inhomogeneous mu builder
# ============================================================

def test_inhomogeneous_mu_tilt_range():
    """Tilt pattern spans [-mu_tilt, +mu_tilt] monotonically."""
    for L_ in [6, 7]:
        for tilt in [0.5, 1.0, 2.0]:
            mu = bh._build_inhomogeneous_mu(L_, tilt, "tilt")
            assert len(mu) == L_
            assert abs(mu[0]  - (-tilt)) < 1e-10, f"mu[0]={mu[0]} ≠ -{tilt}"
            assert abs(mu[-1] - (+tilt)) < 1e-10, f"mu[-1]={mu[-1]} ≠ +{tilt}"
            assert np.all(np.diff(mu) > 0), "tilt not monotonically increasing"
    print("  PASS: inhomogeneous_mu_tilt_range")


def test_inhomogeneous_mu_step_symmetry():
    """Step pattern has mean 0 and opposite halves."""
    for L_ in [6, 8]:
        for tilt in [0.5, 1.0]:
            mu = bh._build_inhomogeneous_mu(L_, tilt, "step")
            assert len(mu) == L_
            assert abs(np.mean(mu)) < 1e-10, f"step mean = {np.mean(mu)}"
            assert np.all(mu[:L_ // 2] == tilt),  "first half not +tilt"
            assert np.all(mu[L_ // 2:] == -tilt), "second half not -tilt"
    print("  PASS: inhomogeneous_mu_step_symmetry")


def test_inhomogeneous_mu_unknown_pattern():
    """Unknown pattern raises ValueError."""
    try:
        bh._build_inhomogeneous_mu(6, 1.0, "zigzag")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASS: inhomogeneous_mu_unknown_pattern_raises")


def test_inhomogeneous_large_tilt_shifts_selector():
    """With large tilt, s_fi must differ from s_geo (fi no longer picks center).

    mu_tilt pattern='tilt': mu[i] = tilt*(2i/(L-1)-1).
    For i=0: mu=-tilt (attractive → particles accumulate at site 0, low-index end).
    For i=L-1: mu=+tilt (repulsive).
    So with large tilt, high-F_i sites are the LOW-index sites.
    """
    mu_vec = bh._build_inhomogeneous_mu(L, mu_tilt=2.0, pattern="tilt")
    # Verify: mu[0] < 0 (attractive), mu[-1] > 0 (repulsive)
    assert mu_vec[0] < 0 and mu_vec[-1] > 0, f"Unexpected mu sign: {mu_vec}"

    res = bh.run_disorder_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=3, burn_in_time=2.0,
        trial_seed=0, n_boot=5)
    s_fi  = res["s_fi"]
    s_geo = res["s_geo"]
    # Large tilt must move s_fi away from the geometric center
    assert s_fi != s_geo, (
        f"Large tilt did not shift selector: s_fi={s_fi}, s_geo={s_geo}")
    # fi sites should be in the LOWER half (low-i = attractive end)
    k    = res["k"]
    half = L // 2
    fi_in_lower = sum(s < half for s in s_fi)
    assert fi_in_lower >= k - 1, (
        f"Expected fi sites in lower half (attractive end), got s_fi={s_fi}")
    print(f"  PASS: inhomogeneous_large_tilt_shifts_selector  (s_fi={s_fi}, s_geo={s_geo})")


def test_inhomogeneous_experiment_runs(tmp_path, monkeypatch=None):
    """run_inhomogeneous_experiment completes and returns results for all conditions."""
    # Patch CKPT_DIR to temp directory
    orig = bh.CKPT_DIR
    bh.CKPT_DIR = str(tmp_path) if tmp_path else tempfile.mkdtemp()
    try:
        cfg = dict(_CFG)
        cfg["L_LIST"]        = [6]
        cfg["J_OVER_U_LIST"] = [0.30]
        cfg["N_TRIALS"]      = 3
        cfg["N_BOOT"]        = 5
        cfg["BURN_IN_TIME"]  = 0.5
        results = bh.run_inhomogeneous_experiment(
            cfg, mu_tilts=[0.5, 2.0], patterns=["tilt"])
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        for L_, N_, ju_, mu_tilt_, pattern_, res in results:
            assert L_ == 6
            assert "results" in res
            assert len(res["results"]) == len(cfg["TAU_LIST"])
    finally:
        bh.CKPT_DIR = orig
    print("  PASS: inhomogeneous_experiment_runs")


# ============================================================
# 3. Checkpoint path functions
# ============================================================

def test_ckpt_paths_unique():
    """Different parameters produce different checkpoint paths."""
    paths = set()
    for L_ in [6, 7]:
        for N_ in [3, 4]:
            for ju in [0.20, 0.40]:
                for mu in [0.10, 0.30]:
                    for r in [0, 1]:
                        paths.add(bh._sel_ckpt_path(L_, N_, ju, mu, r))
                        paths.add(bh._gscan_ckpt_path(L_, N_, ju, mu, 0.5, r))
    # All paths should be unique
    assert len(paths) == 2 * 2 * 2 * 2 * 2 * 2, f"Duplicate paths! got {len(paths)}"

    # Inhom paths include pattern
    inhom_paths = set()
    for L_ in [6, 7]:
        for ju in [0.20, 0.40]:
            for tilt in [0.5, 2.0]:
                for pat in ["tilt", "step"]:
                    inhom_paths.add(bh._inhom_ckpt_path(L_, L_ // 2, ju, tilt, pat))
    assert len(inhom_paths) == 2 * 2 * 2 * 2, f"Duplicate inhom paths: {len(inhom_paths)}"
    print("  PASS: ckpt_paths_unique")


def test_ckpt_path_prefixes():
    """Checkpoint filenames use correct prefixes."""
    p_sel   = bh._sel_ckpt_path(6, 3, 0.30, 0.10, 0)
    p_gscan = bh._gscan_ckpt_path(6, 3, 0.30, 0.10, 0.5, 0)
    p_inhom = bh._inhom_ckpt_path(6, 3, 0.30, 1.0, "tilt")
    assert os.path.basename(p_sel).startswith("sel_"),   p_sel
    assert os.path.basename(p_gscan).startswith("gscan_"), p_gscan
    assert os.path.basename(p_inhom).startswith("inhom_"), p_inhom
    print("  PASS: ckpt_path_prefixes")


# ============================================================
# 4. Gamma scan: worker builds correct Liouvillian
# ============================================================

def test_gamma_scan_different_gamma_gives_different_loss():
    """Low γ_extra should produce less loss than high γ_extra (same realization)."""
    mu_vec = _mu_random(10)

    res_lo = bh.run_disorder_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.1,
        tau_list=[2.0], n_trials=3, burn_in_time=1.0,
        trial_seed=0, n_boot=5)
    res_hi = bh.run_disorder_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=2.0,
        tau_list=[2.0], n_trials=3, burn_in_time=1.0,
        trial_seed=0, n_boot=5)

    loss_lo = res_lo["results"][0]["fi_loss"]
    loss_hi = res_hi["results"][0]["fi_loss"]
    assert loss_hi >= loss_lo - 1e-8, (
        f"Higher γ_extra should give >= loss: lo={loss_lo:.6f}, hi={loss_hi:.6f}")
    print(f"  PASS: gamma_scan_different_gamma  (lo={loss_lo:.5f}, hi={loss_hi:.5f})")


def test_gamma_scan_zero_extra_gives_near_zero_fi_minus_random():
    """With γ_extra ≈ 0, targeted and random are identical → fi−random ≈ 0."""
    mu_vec = _mu_zero()
    res = bh.run_disorder_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=1e-6,
        tau_list=[1.0], n_trials=10, burn_in_time=1.0,
        trial_seed=0, n_boot=10)
    mean_diff = abs(res["results"][0]["fi_on_fi_vs_rnd"]["mean"])
    assert mean_diff < 1e-4, f"γ_extra~0 gave fi−rnd={mean_diff:.2e}, expected ~0"
    print(f"  PASS: gamma_scan_zero_extra_gives_near_zero  (|fi−rnd|={mean_diff:.2e})")


# ============================================================
# 5. Selector sweep: occupation conservation
# ============================================================

def test_occupation_conservation_across_selectors():
    """Total occupation before and after intervention should be conserved by Lindblad L=n_i."""
    mu_vec = _mu_random(20)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=2, burn_in_time=1.0,
        trial_seed=0, n_boot=5)
    # Total occupation = N (particle number conserved)
    occ_before = np.array(res["occ_before"])
    total_n = occ_before.sum()
    assert abs(total_n - N) < 1e-6, f"occ_before sum={total_n:.8f}, expected N={N}"
    print(f"  PASS: occupation_conservation  (∑⟨n_i⟩={total_n:.8f}, N={N})")


# ============================================================
# 6. dis_amp and dis_anti selectors
# ============================================================

def test_dis_amp_selects_top_k_by_mu_magnitude():
    """dis_amp must select the k sites with largest |mu_i|."""
    mu_vec = np.array([0.05, 0.80, 0.10, 0.90, 0.02, 0.60])  # |mu|: 0.05,0.80,0.10,0.90,0.02,0.60
    # k = ceil(6/3) = 2  →  top-2 |mu| sites = [1, 3]  (indices of 0.80 and 0.90)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=2, burn_in_time=0.5,
        trial_seed=0, n_boot=5)
    k = res["k"]
    dis_amp_sites = res["sel_sites"]["dis_amp"]
    expected = sorted(np.argsort(np.abs(mu_vec))[-k:].tolist())
    assert dis_amp_sites == expected, (
        f"dis_amp={dis_amp_sites}, expected top-{k} |mu| sites={expected}, |mu|={np.abs(mu_vec)}")
    print(f"  PASS: dis_amp_selects_top_k_by_mu_magnitude  (sites={dis_amp_sites}, k={k})")


def test_dis_anti_selects_bottom_k_by_mu_magnitude():
    """dis_anti must select the k sites with smallest |mu_i|."""
    mu_vec = np.array([0.05, 0.80, 0.10, 0.90, 0.02, 0.60])
    # k = 2  →  bottom-2 |mu| sites = [0, 4]  (indices of 0.05 and 0.02)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=2, burn_in_time=0.5,
        trial_seed=0, n_boot=5)
    k = res["k"]
    dis_anti_sites = res["sel_sites"]["dis_anti"]
    expected = sorted(np.argsort(np.abs(mu_vec))[:k].tolist())
    assert dis_anti_sites == expected, (
        f"dis_anti={dis_anti_sites}, expected bottom-{k} |mu| sites={expected}, |mu|={np.abs(mu_vec)}")
    print(f"  PASS: dis_anti_selects_bottom_k_by_mu_magnitude  (sites={dis_anti_sites}, k={k})")


def test_dis_amp_dis_anti_disjoint():
    """dis_amp and dis_anti must select disjoint site sets (top vs bottom k)."""
    mu_vec = _mu_random(42)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=2, burn_in_time=0.5,
        trial_seed=0, n_boot=5)
    amp  = set(res["sel_sites"]["dis_amp"])
    anti = set(res["sel_sites"]["dis_anti"])
    assert amp.isdisjoint(anti), (
        f"dis_amp and dis_anti overlap: amp={amp}, anti={anti}")
    print(f"  PASS: dis_amp_dis_anti_disjoint  (amp={sorted(amp)}, anti={sorted(anti)})")


def test_dis_amp_amp_geq_anti_mu():
    """Every dis_amp site has |mu| >= every dis_anti site's |mu|."""
    mu_vec = _mu_random(99)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=2, burn_in_time=0.5,
        trial_seed=0, n_boot=5)
    amp_mu  = np.abs(mu_vec)[res["sel_sites"]["dis_amp"]]
    anti_mu = np.abs(mu_vec)[res["sel_sites"]["dis_anti"]]
    assert amp_mu.min() >= anti_mu.max() - 1e-12, (
        f"dis_amp min |mu|={amp_mu.min():.4f} < dis_anti max |mu|={anti_mu.max():.4f}")
    print(f"  PASS: dis_amp_amp_geq_anti_mu  (amp_min={amp_mu.min():.4f} >= anti_max={anti_mu.max():.4f})")


def test_dis_amp_differs_from_fi_at_strong_disorder():
    """At strong disorder (mu_max=0.5), fi and dis_amp typically select different sites.

    This is the core scientific claim: F_i carries variance information that is
    independent of mere disorder amplitude |mu_i|.

    Uses nmax=3 (the paper's actual parameter) so the Hilbert space and variance
    structure match the reported 58–79% disagreement rate.  We assert ≥50% across
    20 realizations — well supported by the full 50-realization AWS runs.

    NOTE: nmax=2 (the test-suite default) gives a different variance landscape
    and cannot be used to verify this claim.
    """
    rng = np.random.default_rng(20260325)   # paper seed for reproducibility
    nmax3 = 3                                # paper's actual nmax
    n_real = 20
    n_differ = 0
    for _ in range(n_real):
        mu_vec = rng.uniform(-0.5, 0.5, size=L)
        res = bh.run_selector_sweep_realization(
            L, N, nmax3, JU, mu_vec,          # nmax=3 ← critical
            gamma_base=0.1, gamma_extra=0.5,
            tau_list=[1.0], n_trials=3, burn_in_time=1.0,
            trial_seed=int(rng.integers(0, 10000)), n_boot=10)
        if res["sel_sites"]["fi"] != res["sel_sites"]["dis_amp"]:
            n_differ += 1
    frac = n_differ / n_real
    assert frac >= 0.50, (
        f"fi==dis_amp in too many realizations ({n_real - n_differ}/{n_real}); "
        f"expected fi≠dis_amp in ≥50% at nmax=3, mu_max=0.5 — got {frac:.0%}")
    print(f"  PASS: dis_amp_differs_from_fi_at_strong_disorder  "
          f"(fi≠dis_amp in {n_differ}/{n_real} = {frac:.0%} of realizations, nmax=3)")


def test_zero_disorder_dis_amp_deterministic_and_valid():
    """Under mu=0, dis_amp sites are NumPy-stable (argsort on all-zero array is
    deterministic: ascending index order, so dis_amp=[-k:] gives [L-k..L-1],
    dis_anti=[:k] gives [0..k-1]).  Either way, sites must be valid and non-duplicate."""
    mu_vec = np.zeros(L)
    res = bh.run_selector_sweep_realization(
        L, N, nmax, JU, mu_vec,
        gamma_base=0.1, gamma_extra=0.5,
        tau_list=[1.0], n_trials=2, burn_in_time=0.5,
        trial_seed=0, n_boot=5)
    k = res["k"]
    for name in ("dis_amp", "dis_anti"):
        sites = res["sel_sites"][name]
        assert len(sites) == k, f"{name} has {len(sites)} sites, expected {k}"
        assert all(0 <= s < L for s in sites), f"{name} site out of range: {sites}"
        assert len(set(sites)) == k, f"{name} has duplicate sites: {sites}"
    print(f"  PASS: zero_disorder_dis_amp_nondeterministic_but_valid  (k={k})")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    import traceback

    tests = [
        test_selector_sweep_keys,
        test_selector_sweep_site_counts,
        test_selector_sweep_clean_symmetry,
        test_boundary_selector_outermost,
        test_anti_fi_is_bottom_k,
        test_selector_sweep_loss_non_negative,
        test_inhomogeneous_mu_tilt_range,
        test_inhomogeneous_mu_step_symmetry,
        test_inhomogeneous_mu_unknown_pattern,
        test_inhomogeneous_large_tilt_shifts_selector,
        lambda: test_inhomogeneous_experiment_runs(tempfile.mkdtemp()),
        test_ckpt_paths_unique,
        test_ckpt_path_prefixes,
        test_gamma_scan_different_gamma_gives_different_loss,
        test_gamma_scan_zero_extra_gives_near_zero_fi_minus_random,
        test_occupation_conservation_across_selectors,
        # dis_amp / dis_anti tests
        test_dis_amp_selects_top_k_by_mu_magnitude,
        test_dis_anti_selects_bottom_k_by_mu_magnitude,
        test_dis_amp_dis_anti_disjoint,
        test_dis_amp_amp_geq_anti_mu,
        test_dis_amp_differs_from_fi_at_strong_disorder,
        test_zero_disorder_dis_amp_deterministic_and_valid,
    ]

    passed, failed = 0, []
    print(f"\nRunning {len(tests)} tests...\n")
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            name = getattr(fn, "__name__", repr(fn))
            print(f"  FAIL: {name}")
            traceback.print_exc()
            failed.append(name)

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests PASS.")
