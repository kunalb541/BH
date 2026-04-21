#!/usr/bin/env bash
# Extended campaign — full sel sweep with dis_amp/dis_anti, inhomogeneous,
# gamma scan, and L=7 robustness.  EBS persists; self-terminates after S3 sync.
set -euo pipefail
cd /home/ubuntu/BH
PY=/home/ubuntu/venv/bin/python
LOG=outputs
mkdir -p $LOG

# ─── Guarantee S3 sync + self-terminate even if a phase crashes ──────────────
# trap fires on EXIT (normal completion, error, or signal).
# This ensures the instance never hangs billing forever if bh.py crashes.
_cleanup() {
    EXIT_CODE=$?
    echo "=== Cleanup triggered (exit code ${EXIT_CODE}) at $(date) ===" | tee -a $LOG/campaign_done.txt
    echo "=== Syncing checkpoints to S3 ===" | tee -a $LOG/campaign_done.txt
    aws s3 sync outputs/checkpoints/ s3://bh-results-kunal-2026/checkpoints/ \
        --region eu-central-1 2>&1 | tail -5 | tee -a $LOG/campaign_done.txt
    echo "=== Syncing logs/outputs to S3 ===" | tee -a $LOG/campaign_done.txt
    aws s3 sync outputs/ s3://bh-results-kunal-2026/outputs/ \
        --exclude "checkpoints/*" --region eu-central-1 2>&1 | tee -a $LOG/campaign_done.txt
    INSTANCE_ID="$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
    echo "Self-terminating $INSTANCE_ID at $(date)" | tee -a $LOG/campaign_done.txt
    aws ec2 terminate-instances --region eu-central-1 --instance-ids "$INSTANCE_ID"
}
trap _cleanup EXIT

# ─── Shared argument bundles ─────────────────────────────────────────────────
L6="--l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3"
L7="--l-list 7 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3"
DIS_COMMON="--disorder-realizations 50 --dis-workers 64 --resume"

# ─── Phase 1: L=6 selector sweep, ALL disorder strengths ─────────────────────
# CRITICAL: regenerates sel_ checkpoints with dis_amp + dis_anti selectors.
# Old sel_L6_ checkpoints were deleted before rsync (they lacked dis_amp).
# 3 JU × 6 μ × 50 realizations = 900 tasks → ~20 min at 64 workers
echo "=== Phase 1: Selector sweep L=6, all mu (dis_amp/dis_anti) ===" | tee $LOG/phase1.log
$PY bh.py --selector-sweep $L6 $DIS_COMMON \
    --disorder-strengths 0.10 0.20 0.30 0.50 1.00 2.00 \
    >> $LOG/phase1.log 2>&1
echo "Phase 1 done at $(date)" | tee -a $LOG/phase1.log

# ─── Phase 2: L=6 inhomogeneous chain ────────────────────────────────────────
# Deterministic (non-random) asymmetry — single realization per condition.
# 3 JU × 3 tilts × 2 patterns = 18 serial conditions → ~20 min
echo "=== Phase 2: Inhomogeneous chain L=6 ===" | tee $LOG/phase2.log
$PY bh.py --inhomogeneous $L6 \
    --inhom-tilts 0.5 1.0 2.0 --inhom-patterns tilt step --resume \
    >> $LOG/phase2.log 2>&1
echo "Phase 2 done at $(date)" | tee -a $LOG/phase2.log

# ─── Phase 3: L=6 gamma scan ─────────────────────────────────────────────────
# Tests robustness of fi>geo to choice of gamma_extra.
# 2 JU × 1 μ × 5 γ × 50 realizations = 500 tasks → ~15 min at 64 workers
echo "=== Phase 3: Gamma scan L=6 (JU=0.30,0.40  mu=0.10  5 gamma values) ===" | tee $LOG/phase3.log
$PY bh.py --gamma-scan $L6 $DIS_COMMON \
    --disorder-strengths 0.10 \
    --gamma-scan-values 0.1 0.2 0.5 1.0 2.0 \
    >> $LOG/phase3.log 2>&1
echo "Phase 3 done at $(date)" | tee -a $LOG/phase3.log

# ─── Phase 4: L=7 disorder, all mu ───────────────────────────────────────────
# Finite-size extension of disorder experiment to L=7.
# 3 JU × 6 μ × 50 realizations = 900 tasks → ~25 min at 64 workers (L=7 slower)
echo "=== Phase 4: Disorder L=7, all mu ===" | tee $LOG/phase4.log
$PY bh.py --disorder $L7 $DIS_COMMON \
    --disorder-strengths 0.10 0.20 0.30 0.50 1.00 2.00 \
    >> $LOG/phase4.log 2>&1
echo "Phase 4 done at $(date)" | tee -a $LOG/phase4.log

# ─── Phase 5: L=7 shell-perm ─────────────────────────────────────────────────
# Requires Phase 4 checkpoints.  Tests whether fi sites are special beyond
# shell structure at L=7.
echo "=== Phase 5: Shell-perm L=7, all mu ===" | tee $LOG/phase5.log
$PY bh.py --shell-perm $L7 $DIS_COMMON \
    --disorder-strengths 0.10 0.20 0.30 0.50 1.00 2.00 \
    >> $LOG/phase5.log 2>&1
echo "Phase 5 done at $(date)" | tee -a $LOG/phase5.log

# ─── Phase 6: L=7 selector sweep, strong disorder ────────────────────────────
# Key finite-size robustness check: does fi>geo persist at L=7?
# 3 JU × 3 μ × 50 realizations = 450 tasks → ~15 min at 64 workers
echo "=== Phase 6: Selector sweep L=7, strong disorder (mu=0.50,1.00,2.00) ===" | tee $LOG/phase6.log
$PY bh.py --selector-sweep $L7 $DIS_COMMON \
    --disorder-strengths 0.50 1.00 2.00 \
    >> $LOG/phase6.log 2>&1
echo "Phase 6 done at $(date)" | tee -a $LOG/phase6.log

echo "=== All phases complete at $(date) ===" | tee $LOG/campaign_done.txt
# _cleanup trap fires automatically on normal exit — S3 sync + self-terminate handled there.
