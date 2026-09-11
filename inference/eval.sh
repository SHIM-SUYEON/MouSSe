#!/usr/bin/env bash
# eval.sh
#
# Comprehensive evaluation for the MouSSe single-stage PIT+UQ model.
# Experiments run in parallel pairs using 4 GPUs (0,1) + (2,3).
# Run from the repository root, e.g.  bash inference/eval.sh main
#
# Usage:
#   bash inference/eval.sh               # full suite
#   bash inference/eval.sh main          # backbone + adaptive only (cap=1300)
#   bash inference/eval.sh lambda        # fixed-lambda sweep
#   bash inference/eval.sh adaptive      # adaptive-lambda sweep
#   bash inference/eval.sh steps         # ODE steps sweep
#   bash inference/eval.sh sigma         # STO sigma analysis only
#   bash inference/eval.sh summary       # print summary tables from existing logs

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT="$(dirname "$0")/eval.py"
CKPT="/shared/NAS_SSD/syshim/hi-d/bluejays/speechflowuq/supervoice_separate/racoon-aug/single-stage/0826/large-sep-0820.best.pt"
# CKPT="/shared/NAS_SSD/syshim/hi-d/bluejays/speechflowuq/supervoice_separate/racoon-aug/single-stage/0829-lambda1/large-sep-0820.62000.pt"
OUT_BASE="results/stage1_uq_0826"
LOG_DIR="logs/stage1_uq_1300_0826"

# Checkpoint trained WITHOUT entropy weighting (needed for entropy ablation).
# Export CKPT_NO_ENTROPY=/path/to/ckpt.pt to enable that section; otherwise it is skipped.
CKPT_NO_ENTROPY="${CKPT_NO_ENTROPY:-}"

# ── Config ─────────────────────────────────────────────────────────────────────
GPU_A="0"   # first GPU
GPU_B="1"   # second GPU (run in parallel with GPU_A)
CAP_MAIN=1300  # full test set for final reported numbers
CAP_SWEEP=1300  # quick sweep (300 is enough to find the optimum)

mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
_run() {
    # _run <log> <extra args...>
    local log="$1"; shift
    echo "[RUN] $* → $log"
    python3 -u "$SCRIPT" "$@" > "$log" 2>&1
}

# Run two experiments in parallel; wait for both.
_pair() {
    local log_a="$1"; shift
    local args_a=("$@")
    # find the split between A and B args (separated by --SEP--)
    local idx=0
    local a_args=() b_args=()
    local in_b=false
    for arg in "${args_a[@]}"; do
        if [[ "$arg" == "--SEP--" ]]; then in_b=true; continue; fi
        if $in_b; then b_args+=("$arg"); else a_args+=("$arg"); fi
    done
    echo "[PAIR] starting two jobs in parallel"
    _run "${a_args[0]}" "${a_args[@]:1}" &
    PID_A=$!
    _run "${b_args[0]}" "${b_args[@]:1}" &
    PID_B=$!
    wait $PID_A $PID_B
    echo "[PAIR] both done"
}

run_lambda() {
    local lam="$1" gpu="$2" cap="${3:-$CAP_SWEEP}"
    local tag; tag=$(echo "$lam" | tr '.' '_')
    local log="$LOG_DIR/lambda_fixed_${tag}.log"
    local out="$OUT_BASE/lambda_fixed_${tag}"
    _run "$log" --ckpt "$CKPT" --stage 1 --lambda-fixed "$lam" \
        --gpu "$gpu" --cap "$cap" --out "$out"
}

run_adaptive() {
    local scale="$1" gpu="$2" cap="${3:-$CAP_SWEEP}"
    local tag; tag=$(echo "$scale" | tr '.' '_')
    local log="$LOG_DIR/adaptive_scale_${tag}.log"
    local out="$OUT_BASE/adaptive_scale_${tag}"
    _run "$log" --ckpt "$CKPT" --stage 2 \
        --lambda-scale "$scale" --gpu "$gpu" --cap "$cap" --out "$out"
}

run_steps() {
    local steps="$1" scale="$2" gpu="$3" cap="${4:-$CAP_SWEEP}"
    local log="$LOG_DIR/steps_${steps}.log"
    local out="$OUT_BASE/steps_${steps}"
    _run "$log" --ckpt "$CKPT" --stage 2 \
        --lambda-scale "$scale" --steps "$steps" \
        --gpu "$gpu" --cap "$cap" --out "$out"
}

# ── Section 0: Main comparison (full cap=1300 for paper numbers) ───────────────
run_main() {
    echo ""
    echo "======================================================================"
    echo " MAIN  (CAP=$CAP_MAIN — for paper numbers)"
    echo "======================================================================"

    # Backbone baseline (stage 1, fixed alpha=1.5, no UQ guidance)
    LOG_BB="$LOG_DIR/main_backbone.log"
    _run "$LOG_BB" \
        --ckpt "$CKPT" --stage 1 \
        --gpu "$GPU_A" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/main_backbone" &
    PID_A=$!

    # Adaptive UQ guidance (lambda_scale=1.5 → lambda* ≈ 0.50)
    LOG_AD="$LOG_DIR/main_adaptive_1_5.log"
    _run "$LOG_AD" \
        --ckpt "$CKPT" --stage 2 --lambda-scale 1.5 \
        --gpu "$GPU_B" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/main_adaptive_1_5" &
    PID_B=$!

    wait $PID_A $PID_B
    echo "[main] backbone + adaptive_1.5 done."

    # Fixed lambda at near-optimal (lambda=0.5, same average as adaptive_1.5)
    LOG_FX="$LOG_DIR/main_fixed_0_5.log"
    _run "$LOG_FX" \
        --ckpt "$CKPT" --stage 1 --lambda-fixed 0.5 \
        --gpu "$GPU_A" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/main_fixed_0_5"
    echo "[main] fixed_0.5 done."
}

# ── Section 1: Fixed-lambda sweep ──────────────────────────────────────────────
run_lambda_sweep() {
    echo ""
    echo "======================================================================"
    echo " FIXED-LAMBDA SWEEP  (CAP=$CAP_SWEEP)"
    echo "======================================================================"

    run_lambda 0.0  "$GPU_A" "$CAP_SWEEP" &
    run_lambda 0.25 "$GPU_B" "$CAP_SWEEP" &
    wait

    run_lambda 0.5  "$GPU_A" "$CAP_SWEEP" &
    run_lambda 0.75 "$GPU_B" "$CAP_SWEEP" &
    wait

    run_lambda 1.0  "$GPU_A" "$CAP_SWEEP" &
    wait

    echo "[sweep] Fixed-lambda sweep complete."
}

# ── Section 2: Adaptive-lambda (lambda_scale) sweep ───────────────────────────
run_adaptive_sweep() {
    echo ""
    echo "======================================================================"
    echo " ADAPTIVE LAMBDA_SCALE SWEEP  (CAP=$CAP_SWEEP)"
    echo "======================================================================"

    # scale 0.0 = no UQ guidance (backbone + alpha=1.5 CFG, stage 1)
    _run "$LOG_DIR/adaptive_scale_0_0.log" \
        --ckpt "$CKPT" --stage 1 \
        --gpu "$GPU_A" --cap "$CAP_SWEEP" --out "$OUT_BASE/adaptive_scale_0_0" &
    run_adaptive 0.25 "$GPU_B" "$CAP_SWEEP" &
    wait

    run_adaptive 0.5  "$GPU_A" "$CAP_SWEEP" &
    run_adaptive 0.75 "$GPU_B" "$CAP_SWEEP" &
    wait

    run_adaptive 1.0  "$GPU_A" "$CAP_SWEEP" &
    run_adaptive 1.25 "$GPU_B" "$CAP_SWEEP" &
    wait

    run_adaptive 1.5  "$GPU_A" "$CAP_SWEEP" &
    run_adaptive 1.75 "$GPU_B" "$CAP_SWEEP" &
    wait

    run_adaptive 2.0  "$GPU_A" "$CAP_SWEEP" &
    wait

    echo "[sweep] Adaptive lambda_scale sweep complete."
}

# ── Section 3: ODE steps sweep ────────────────────────────────────────────────
run_steps_sweep() {
    local best_scale="${1:-1.5}"
    echo ""
    echo "======================================================================"
    echo " ODE STEPS SWEEP  (adaptive scale=$best_scale, CAP=$CAP_SWEEP)"
    echo "======================================================================"

    run_steps 1  "$best_scale" "$GPU_A" "$CAP_SWEEP" &
    run_steps 2  "$best_scale" "$GPU_B" "$CAP_SWEEP" &
    wait

    run_steps 4  "$best_scale" "$GPU_A" "$CAP_SWEEP" &
    run_steps 8  "$best_scale" "$GPU_B" "$CAP_SWEEP" &
    wait

    run_steps 16 "$best_scale" "$GPU_A" "$CAP_SWEEP" &
    run_steps 32 "$best_scale" "$GPU_B" "$CAP_SWEEP" &
    wait

    run_steps 64 "$best_scale" "$GPU_A" "$CAP_SWEEP" &
    run_steps 128 "$best_scale" "$GPU_B" "$CAP_SWEEP" &
    wait

    echo "[sweep] Steps sweep complete."
}

# ── Section 4: Entropy-weighted training ablation ─────────────────────────────
run_entropy_ablation() {
    if [[ -z "$CKPT_NO_ENTROPY" ]]; then
        echo "[skip] CKPT_NO_ENTROPY not set — skipping entropy ablation."
        return
    fi
    echo ""
    echo "======================================================================"
    echo " ENTROPY ABLATION  (no-entropy checkpoint)"
    echo "======================================================================"

    LOG_NE="$LOG_DIR/no_entropy_backbone.log"
    _run "$LOG_NE" \
        --ckpt "$CKPT_NO_ENTROPY" --stage 1 \
        --gpu "$GPU_A" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/no_entropy_backbone" &
    PID_A=$!

    LOG_NE_AD="$LOG_DIR/no_entropy_adaptive.log"
    _run "$LOG_NE_AD" \
        --ckpt "$CKPT_NO_ENTROPY" --stage 2 --lambda-scale 1.5 \
        --gpu "$GPU_B" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/no_entropy_adaptive" &
    PID_B=$!

    wait $PID_A $PID_B
    echo "[ablation] Entropy ablation done."
}

# ── Section 5: STO sigma analysis (S/T/O region sigma_cond vs sigma_uncond) ───
run_sigma_analysis() {
    echo ""
    echo "======================================================================"
    echo " STO SIGMA ANALYSIS  (sigma_cond vs sigma_uncond by region)"
    echo "======================================================================"
    # Runs with lambda_scale=1.5 so that sigma_cond AND sigma_uncond are computed.
    # STO region breakdown is printed automatically in the eval output.

    LOG_SIG="$LOG_DIR/sigma_sto_analysis.log"
    _run "$LOG_SIG" \
        --ckpt "$CKPT" --stage 2 --lambda-scale 1.5 \
        --gpu "$GPU_A" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/sigma_sto_analysis"

    echo "[sigma] STO sigma analysis done. See $LOG_SIG for [STO-sigma] lines."
}

# ── Section 6: Summary ────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo "======================================================================"
    echo " SUMMARY"
    echo "======================================================================"

    echo ""
    echo "--- Fixed-lambda sweep (SI-SNRi | SI-SDRi(vo) per n_src) ---"
    printf "%-8s | %12s %12s %12s | %12s %12s %12s\n" \
           "lambda" "SI-SNRi 2spk" "SI-SNRi 3spk" "SI-SNRi 4spk" \
           "SI-SDRi 2spk" "SI-SDRi 3spk" "SI-SDRi 4spk"
    printf "%-8s-+-%12s-%12s-%12s-+-%12s-%12s-%12s\n" \
           "--------" "------------" "------------" "------------" \
           "------------" "------------" "------------"
    for lam in 0.0 0.25 0.5 0.75 1.0 1.5; do
        local tag; tag=$(echo "$lam" | tr '.' '_')
        local log="$LOG_DIR/lambda_fixed_${tag}.log"
        if [[ -f "$log" ]]; then
            s2=$(grep "\[n_src=2\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            s3=$(grep "\[n_src=3\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            s4=$(grep "\[n_src=4\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            d2=$(grep "\[n_src=2\]" "$log" | grep -oP "SI-SDRi\(vo\)=\K[0-9.-]+" | tail -1 || echo "---")
            d3=$(grep "\[n_src=3\]" "$log" | grep -oP "SI-SDRi\(vo\)=\K[0-9.-]+" | tail -1 || echo "---")
            d4=$(grep "\[n_src=4\]" "$log" | grep -oP "SI-SDRi\(vo\)=\K[0-9.-]+" | tail -1 || echo "---")
            printf "%-8s | %12s %12s %12s | %12s %12s %12s\n" "$lam" "$s2" "$s3" "$s4" "$d2" "$d3" "$d4"
        else
            printf "%-8s | (not run)\n" "$lam"
        fi
    done

    echo ""
    echo "--- Adaptive lambda_scale sweep (SI-SNRi | SI-SDRi(vo) per n_src) ---"
    printf "%-8s | %12s %12s %12s | %12s %12s %12s\n" \
           "scale" "SI-SNRi 2spk" "SI-SNRi 3spk" "SI-SNRi 4spk" \
           "SI-SDRi 2spk" "SI-SDRi 3spk" "SI-SDRi 4spk"
    printf "%-8s-+-%12s-%12s-%12s-+-%12s-%12s-%12s\n" \
           "--------" "------------" "------------" "------------" \
           "------------" "------------" "------------"
    for scale in 0.5 1.0 1.5 2.0; do
        local tag; tag=$(echo "$scale" | tr '.' '_')
        local log="$LOG_DIR/adaptive_scale_${tag}.log"
        if [[ -f "$log" ]]; then
            s2=$(grep "\[n_src=2\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            s3=$(grep "\[n_src=3\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            s4=$(grep "\[n_src=4\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            d2=$(grep "\[n_src=2\]" "$log" | grep -oP "SI-SDRi\(vo\)=\K[0-9.-]+" | tail -1 || echo "---")
            d3=$(grep "\[n_src=3\]" "$log" | grep -oP "SI-SDRi\(vo\)=\K[0-9.-]+" | tail -1 || echo "---")
            d4=$(grep "\[n_src=4\]" "$log" | grep -oP "SI-SDRi\(vo\)=\K[0-9.-]+" | tail -1 || echo "---")
            printf "%-8s | %12s %12s %12s | %12s %12s %12s\n" "$scale" "$s2" "$s3" "$s4" "$d2" "$d3" "$d4"
        else
            printf "%-8s | (not run)\n" "$scale"
        fi
    done

    echo ""
    echo "--- ODE Steps sweep (SI-SNRi per n_src) ---"
    printf "%-8s | %12s %12s %12s\n" "steps" "SI-SNRi 2spk" "SI-SNRi 3spk" "SI-SNRi 4spk"
    printf "%-8s-+-%12s-%12s-%12s\n" "--------" "------------" "------------" "------------"
    for steps in 4 8 16 32; do
        local log="$LOG_DIR/steps_${steps}.log"
        if [[ -f "$log" ]]; then
            s2=$(grep "\[n_src=2\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            s3=$(grep "\[n_src=3\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            s4=$(grep "\[n_src=4\]" "$log" | grep -oP "SI-SNRi=\K[0-9.-]+" | tail -1 || echo "---")
            printf "%-8s | %12s %12s %12s\n" "$steps" "$s2" "$s3" "$s4"
        else
            printf "%-8s | (not run)\n" "$steps"
        fi
    done

    echo ""
    echo "--- Main comparison (cap=$CAP_MAIN) ---"
    for tag in main_backbone main_fixed_0_5 main_adaptive_1_5 no_entropy_backbone no_entropy_adaptive; do
        local log="$LOG_DIR/${tag}.log"
        if [[ -f "$log" ]]; then
            overall=$(grep "^\[OVERALL\]" "$log" | tail -1 || echo "(no OVERALL line yet)")
            echo "  [$tag] $overall"
        fi
    done

    echo ""
    echo "--- STO sigma analysis ---"
    local sig_log="$LOG_DIR/sigma_sto_analysis.log"
    if [[ -f "$sig_log" ]]; then
        grep "STO-sigma" "$sig_log" || echo "  (no STO-sigma lines found)"
    else
        echo "  (not run yet)"
    fi
}

# ── Entry point ───────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    main)
        run_main
        print_summary
        ;;
    lambda)
        run_lambda_sweep
        print_summary
        ;;
    adaptive)
        run_adaptive_sweep
        print_summary
        ;;
    steps)
        run_steps_sweep "${2:-1.5}"
        print_summary
        ;;
    sigma)
        run_sigma_analysis
        print_summary
        ;;
    entropy)
        run_entropy_ablation
        print_summary
        ;;
    summary)
        print_summary
        ;;
    all)
        run_main
        run_lambda_sweep
        run_adaptive_sweep
        run_steps_sweep "1.5"
        run_sigma_analysis
        run_entropy_ablation
        print_summary
        ;;
    *)
        echo "Usage: bash $0 [main|lambda|adaptive|steps|sigma|entropy|summary|all]"
        exit 1
        ;;
esac
