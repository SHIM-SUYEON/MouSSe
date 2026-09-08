"""Metric accumulation, category breakdown, and summary reporting for evaluation.

These helpers are inference/eval only (not used by training). They collect
per-sample metrics into accumulators, break them down by language / gender /
overlap categories, print the summary + cross tables, and assemble the final
results dict written to JSON.

The set of speaker counts to report over is passed in explicitly as
``n_src_list`` so this module has no dependency on the eval script's globals.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np


# ─── Category definitions ─────────────────────────────────────────────────────
LANG_CATS = ["EE", "KK", "EK"]
GENDER_CATS = ["MM", "FF", "MF"]
OVERLAP_CATS = ["0", "25", "50", "75", "100"]


# ─── Category accumulator helpers ─────────────────────────────────────────────
def _make_acc():
    return {
        "sisnri": 0.0, "n_sisnri": 0,
        "si_snr_out": 0.0, "n_si_snr_out": 0,       # absolute SI-SNR of separated output (mel)
        "si_snr_mix": 0.0, "n_si_snr_mix": 0,       # SI-SNR of mixture baseline (mel)
        "sisdri_vo":     0.0, "n_sisdri_vo":     0,
        "si_sdr_out_vo": 0.0, "n_si_sdr_out_vo": 0,  # absolute SI-SDR(output, vocoded tgt) — baseline-free
        "si_sdr_mix_vo": 0.0, "n_si_sdr_mix_vo": 0,  # absolute SI-SDR(mixture, vocoded tgt) — shows inflation
        "sdri_vo":       0.0, "n_sdri_vo":       0,
        "sisdri_orig":   0.0, "n_sisdri_orig":   0,
        "sdri_orig":     0.0, "n_sdri_orig":     0,
        "estoi":  0.0, "n_estoi":  0,               # Extended STOI (absolute)
        "estoii": 0.0, "n_estoii": 0,               # ESTOi improvement over mixture baseline
        "stoi":   0.0, "n_stoi":   0,
        "stoii":  0.0, "n_stoii":  0,               # STOI improvement (output − mixture baseline)
        "pesq": 0.0, "n_pesq": 0,
        "stoi_orig":   0.0, "n_stoi_orig":   0,     # STOI vs original wav (paper-standard)
        "stoii_orig":  0.0, "n_stoii_orig":  0,
        "estoi_orig":  0.0, "n_estoi_orig":  0,
        "estoii_orig": 0.0, "n_estoii_orig": 0,
        "pesq_orig":   0.0, "n_pesq_orig":   0,
        "dur": 0.0, "n_dur": 0,
        "snri_vo":  0.0, "n_snri_vo":  0,              # waveform SNRi vs vocoded target
        "pesqi":    0.0, "n_pesqi":    0,              # PESQ improvement over mixture baseline
        "uq": 0.0, "n_uq": 0,
        "uq_active": 0.0, "n_uq_active": 0,         # UQ sigma restricted to active-speech frames
        "uq_silence": 0.0, "n_uq_silence": 0,       # sigma_cond in Silence region (target inactive)
        "uq_target":  0.0, "n_uq_target":  0,       # sigma_cond in Target-only region
        "uq_overlap": 0.0, "n_uq_overlap": 0,       # sigma_cond in Overlap region
        "uq_uncond_silence": 0.0, "n_uq_uncond_silence": 0,  # sigma_uncond in Silence
        "uq_uncond_target":  0.0, "n_uq_uncond_target":  0,  # sigma_uncond in Target-only
        "uq_uncond_overlap": 0.0, "n_uq_uncond_overlap": 0,  # sigma_uncond in Overlap
        "n_samples": 0,
    }


def _acc_add(acc, sisnri=None, si_snr_out=None, si_snr_mix=None,
             sisdri_vo=None, si_sdr_out_vo=None, si_sdr_mix_vo=None,
             sdri_vo=None, snri_vo=None, sisdri_orig=None, sdri_orig=None,
             estoi=None, estoii=None, stoi=None, stoii=None,
             pesq=None, pesqi=None, dur=None, uq=None, uq_active=None,
             uq_silence=None, uq_target=None, uq_overlap=None,
             uq_uncond_silence=None, uq_uncond_target=None, uq_uncond_overlap=None,
             stoi_orig=None, stoii_orig=None, estoi_orig=None,
             estoii_orig=None, pesq_orig=None):
    if sisnri is not None:
        acc["sisnri"]          += sisnri;          acc["n_sisnri"]          += 1
    if si_snr_out is not None:
        acc["si_snr_out"]      += si_snr_out;      acc["n_si_snr_out"]      += 1
    if si_snr_mix is not None:
        acc["si_snr_mix"]      += si_snr_mix;      acc["n_si_snr_mix"]      += 1
    if sisdri_vo is not None:
        acc["sisdri_vo"]       += sisdri_vo;       acc["n_sisdri_vo"]       += 1
    if si_sdr_out_vo is not None:
        acc["si_sdr_out_vo"]   += si_sdr_out_vo;   acc["n_si_sdr_out_vo"]   += 1
    if si_sdr_mix_vo is not None:
        acc["si_sdr_mix_vo"]   += si_sdr_mix_vo;   acc["n_si_sdr_mix_vo"]   += 1
    if sdri_vo is not None:
        acc["sdri_vo"]         += sdri_vo;         acc["n_sdri_vo"]         += 1
    if snri_vo is not None:
        acc["snri_vo"]         += snri_vo;         acc["n_snri_vo"]         += 1
    if sisdri_orig is not None:
        acc["sisdri_orig"]     += sisdri_orig;     acc["n_sisdri_orig"]     += 1
    if sdri_orig is not None:
        acc["sdri_orig"]       += sdri_orig;       acc["n_sdri_orig"]       += 1
    if estoi is not None:
        acc["estoi"]           += estoi;           acc["n_estoi"]           += 1
    if estoii is not None:
        acc["estoii"]          += estoii;          acc["n_estoii"]          += 1
    if stoi is not None:
        acc["stoi"]            += stoi;            acc["n_stoi"]            += 1
    if stoii is not None:
        acc["stoii"]           += stoii;           acc["n_stoii"]           += 1
    if pesq is not None:
        acc["pesq"]            += pesq;            acc["n_pesq"]            += 1
    if pesqi is not None:
        acc["pesqi"]           += pesqi;           acc["n_pesqi"]           += 1
    if dur is not None:
        acc["dur"]             += dur;             acc["n_dur"]             += 1
    if uq is not None:
        acc["uq"]              += uq;              acc["n_uq"]              += 1
    if uq_active is not None:
        acc["uq_active"]       += uq_active;       acc["n_uq_active"]       += 1
    if uq_silence is not None:
        acc["uq_silence"]      += uq_silence;      acc["n_uq_silence"]      += 1
    if uq_target is not None:
        acc["uq_target"]       += uq_target;       acc["n_uq_target"]       += 1
    if uq_overlap is not None:
        acc["uq_overlap"]      += uq_overlap;      acc["n_uq_overlap"]      += 1
    if uq_uncond_silence is not None:
        acc["uq_uncond_silence"] += uq_uncond_silence; acc["n_uq_uncond_silence"] += 1
    if uq_uncond_target is not None:
        acc["uq_uncond_target"]  += uq_uncond_target;  acc["n_uq_uncond_target"]  += 1
    if uq_uncond_overlap is not None:
        acc["uq_uncond_overlap"] += uq_uncond_overlap; acc["n_uq_uncond_overlap"] += 1
    if stoi_orig is not None:
        acc["stoi_orig"]       += stoi_orig;       acc["n_stoi_orig"]       += 1
    if stoii_orig is not None:
        acc["stoii_orig"]      += stoii_orig;      acc["n_stoii_orig"]      += 1
    if estoi_orig is not None:
        acc["estoi_orig"]      += estoi_orig;      acc["n_estoi_orig"]      += 1
    if estoii_orig is not None:
        acc["estoii_orig"]     += estoii_orig;     acc["n_estoii_orig"]     += 1
    if pesq_orig is not None:
        acc["pesq_orig"]       += pesq_orig;       acc["n_pesq_orig"]       += 1


def _merge_acc(dst, src):
    for k in ("sisnri", "si_snr_out", "si_snr_mix",
              "sisdri_vo", "si_sdr_out_vo", "si_sdr_mix_vo",
              "sdri_vo", "snri_vo", "sisdri_orig", "sdri_orig",
              "estoi", "estoii", "stoi", "stoii", "pesq", "pesqi", "dur", "uq", "uq_active",
              "uq_silence", "uq_target", "uq_overlap",
              "uq_uncond_silence", "uq_uncond_target", "uq_uncond_overlap",
              "stoi_orig", "stoii_orig", "estoi_orig", "estoii_orig", "pesq_orig"):
        dst[k] += src[k]
    for k in ("n_sisnri", "n_si_snr_out", "n_si_snr_mix",
              "n_sisdri_vo", "n_si_sdr_out_vo", "n_si_sdr_mix_vo",
              "n_sdri_vo", "n_snri_vo", "n_sisdri_orig", "n_sdri_orig",
              "n_estoi", "n_estoii", "n_stoi", "n_stoii",
              "n_pesq", "n_pesqi", "n_dur", "n_uq", "n_uq_active",
              "n_uq_silence", "n_uq_target", "n_uq_overlap",
              "n_uq_uncond_silence", "n_uq_uncond_target", "n_uq_uncond_overlap",
              "n_stoi_orig", "n_stoii_orig", "n_estoi_orig", "n_estoii_orig", "n_pesq_orig",
              "n_samples"):
        dst[k] += src[k]


def _avg(acc, metric):
    s, n = acc[metric], acc[f"n_{metric}"]
    return s / n if n > 0 else float("nan")


def _load_meta(mix_path):
    """Load JSON metadata (language, gender, overlap, duration) for a mixture."""
    p = Path(mix_path)
    json_path = p.parent.parent / "json" / (p.stem + ".json")
    try:
        with open(json_path) as f:
            d = json.load(f)
        langs = [s["language"] for s in d["speakers"]]  # "ENG" or "KOR"
        genders = [s["gender"] for s in d["speakers"]]  # "Male" or "Female"
        durations = [float(s.get("length_sec", 0.0)) for s in d["speakers"]]
        overlap_pct = int(round(d["overlapped_ratio"] * 100))
        total_length_sec = float(d.get("total_length_sec", sum(durations)))
        return {"langs": langs, "genders": genders, "overlap_pct": overlap_pct,
                "durations": durations, "total_length_sec": total_length_sec}
    except Exception:
        return None


def _categorize(meta):
    """Return (lang_cat, gender_cat, overlap_cat) strings, or None if unknown."""
    if meta is None:
        return None, None, None
    n = len(meta["langs"])
    n_eng = sum(l == "ENG" for l in meta["langs"])
    n_kor = sum(l == "KOR" for l in meta["langs"])
    lang = "EE" if n_eng == n else ("KK" if n_kor == n else "EK")
    n_male = sum(g == "Male" for g in meta["genders"])
    n_fem = sum(g == "Female" for g in meta["genders"])
    gender = "MM" if n_male == n else ("FF" if n_fem == n else "MF")
    overlap = str(meta["overlap_pct"])
    if overlap not in OVERLAP_CATS:
        overlap = None
    return lang, gender, overlap


# ─── Formatting ───────────────────────────────────────────────────────────────
def _fmt_metrics(acc):
    # Paper metrics: improvement over mixture baseline, vs vocoded target.
    dr_vo  = _avg(acc, "sdri_vo")     # SDRi
    sd_vo  = _avg(acc, "sisdri_vo")   # SI-SDRi
    sti    = _avg(acc, "stoii")       # STOIi
    eti    = _avg(acc, "estoii")      # ESTOIi
    pei    = _avg(acc, "pesqi")       # PESQi
    uq     = _avg(acc, "uq")
    uq_act = _avg(acc, "uq_active")
    def _f2(v): return f"{v:6.2f}" if not math.isnan(v) else "  N/A "
    def _f4(v): return f"{v:.4f}"  if not math.isnan(v) else "  N/A"
    def _f6(v): return f"{v:.6f}"  if not math.isnan(v) else "  N/A"
    return (
        f"SDRi(vo)={_f2(dr_vo)} dB  SI-SDRi(vo)={_f2(sd_vo)} dB  "
        f"STOIi={_f4(sti)}  ESTOIi={_f4(eti)}  PESQi={_f4(pei)}  "
        f"UQ(sigma)={_f6(uq)}  UQ(sigma_active)={_f6(uq_act)}"
    )


def _fmt_cell(acc, metric):
    """11-char cell for cross-table."""
    val = _avg(acc, metric)
    if math.isnan(val):
        return f"{'N/A':>11}"
    if metric in ("sdri_vo", "sisdri_vo"):  # dB-valued metrics
        return f"{val:11.2f}"
    return f"{val:11.4f}"


def _print_category_summary(nsrc_acc, cat_acc, n_src_list):
    """Print per-category summary (first table)."""

    def row(cat_type, value, acc):
        n = acc["n_samples"]
        return (f"  {cat_type:<13}[{value:<14}]: n={n:<6}  {_fmt_metrics(acc)}")

    print("\n" + "=" * 90)
    print("  Per-category summary")
    print("=" * 90)

    # n_src rows
    for n in n_src_list:
        print(row("n_src", str(n), nsrc_acc[n]))

    # lang rows (aggregate across all n_src)
    lang_agg = {c: _make_acc() for c in LANG_CATS}
    for n in n_src_list:
        for c in LANG_CATS:
            _merge_acc(lang_agg[c], cat_acc[n]["lang"][c])
    lang_labels = {"EE": "en_only", "KK": "ko_only", "EK": "mixed_lang"}
    for c in LANG_CATS:
        print(row("lang", lang_labels[c], lang_agg[c]))

    # lang_type rows
    mixed_agg = _make_acc()
    single_agg = _make_acc()
    for n in n_src_list:
        _merge_acc(mixed_agg, cat_acc[n]["lang"]["EK"])
        for c in ("EE", "KK"):
            _merge_acc(single_agg, cat_acc[n]["lang"][c])
    print(row("lang_type", "mixed_lang", mixed_agg))
    print(row("lang_type", "single_lang", single_agg))

    # gender mixed_gender row
    mf_agg = _make_acc()
    for n in n_src_list:
        _merge_acc(mf_agg, cat_acc[n]["gender"]["MF"])
    print(row("gender", "mixed_gender", mf_agg))

    # overlap rows (in user-specified order)
    ovlp_agg = {c: _make_acc() for c in OVERLAP_CATS}
    for n in n_src_list:
        for c in OVERLAP_CATS:
            _merge_acc(ovlp_agg[c], cat_acc[n]["overlap"][c])
    for c in ("0", "100", "25", "50", "75"):
        print(row("overlap", f"{c}%", ovlp_agg[c]))


def _print_cross_table(cat_acc, n_src_list):
    """Print cross-tables (one per metric)."""
    METRICS = [
        # Paper metrics: improvement over mixture baseline, vs vocoded target.
        ("sdri_vo",   "SDRi (dB) [wav, improvement over mixture baseline]"),
        ("sisdri_vo", "SI-SDRi (dB) [wav, improvement over mixture baseline]"),
        ("stoii",     "STOIi [STOI improvement over mixture baseline]"),
        ("estoii",    "ESTOIi [Extended STOI improvement over mixture baseline]"),
        ("pesqi",     "PESQi [PESQ improvement over mixture baseline]"),
        # Model uncertainty (not a separation-quality metric)
        ("uq",        "UQ sigma_cond (mean over all frames, per slot)"),
        ("uq_active", "UQ sigma_active (mean over active-speech frames only)"),
    ]

    HDR0 = (
        "       |            Language             "
        "|                 Overlapped Ratio (%)                  "
        "|             Gender              "
    )
    HDR1 = (
        " n_src |         EE         KK         EK"
        "|          0         25         50         75        100"
        "|         MM         FF         MF"
    )
    SEP = (
        "-------+---------------------------------"
        "+-------------------------------------------------------"
        "+---------------------------------"
    )

    for metric, label in METRICS:
        print(f"\n{'=' * len(SEP)}")
        print(f"  {label}")
        print(HDR0)
        print(HDR1)
        print(SEP)
        for n in n_src_list:
            lang_cells = "".join(_fmt_cell(cat_acc[n]["lang"][c], metric) for c in ("EE", "KK", "EK"))
            ovlp_cells = "".join(_fmt_cell(cat_acc[n]["overlap"][c], metric) for c in ("0", "25", "50", "75", "100"))
            gndr_cells = "".join(_fmt_cell(cat_acc[n]["gender"][c], metric) for c in ("MM", "FF", "MF"))
            print(f"{n:>6d} |{lang_cells}|{ovlp_cells}|{gndr_cells}")


def _finalize_results(nsrc_acc, cat_acc, per_sample_records, n_src_list):
    """Print summary tables, build and return the full results dict."""
    _print_category_summary(nsrc_acc, cat_acc, n_src_list)
    _print_cross_table(cat_acc, n_src_list)

    def _jv(v):
        return None if isinstance(v, float) and math.isnan(v) else round(float(v), 6)

    def _acc_to_dict(acc):
        return {
            "n_samples": acc["n_samples"],
            # Paper metrics: improvement over mixture baseline, vs vocoded target.
            "sdri_vo":         _jv(_avg(acc, "sdri_vo")),        "n_sdri_vo":      acc["n_sdri_vo"],
            "si_sdri_vo":      _jv(_avg(acc, "sisdri_vo")),    "n_sisdri_vo":    acc["n_sisdri_vo"],
            "stoii":           _jv(_avg(acc, "stoii")),           "n_stoii":        acc["n_stoii"],
            "estoii":          _jv(_avg(acc, "estoii")),         "n_estoii":       acc["n_estoii"],
            "pesqi":           _jv(_avg(acc, "pesqi")),           "n_pesqi":        acc["n_pesqi"],
            # Model uncertainty
            "uq_sigma":        _jv(_avg(acc, "uq")),              "n_uq":           acc["n_uq"],
            "uq_sigma_active": _jv(_avg(acc, "uq_active")),      "n_uq_active":    acc["n_uq_active"],
            "uq_sigma_silence":        _jv(_avg(acc, "uq_silence")),        "n_uq_silence":        acc["n_uq_silence"],
            "uq_sigma_target":         _jv(_avg(acc, "uq_target")),         "n_uq_target":         acc["n_uq_target"],
            "uq_sigma_overlap":        _jv(_avg(acc, "uq_overlap")),        "n_uq_overlap":        acc["n_uq_overlap"],
            "uq_uncond_silence":       _jv(_avg(acc, "uq_uncond_silence")), "n_uq_uncond_silence": acc["n_uq_uncond_silence"],
            "uq_uncond_target":        _jv(_avg(acc, "uq_uncond_target")),  "n_uq_uncond_target":  acc["n_uq_uncond_target"],
            "uq_uncond_overlap":       _jv(_avg(acc, "uq_uncond_overlap")), "n_uq_uncond_overlap": acc["n_uq_uncond_overlap"],
        }

    results = {"per_nsrc": {}, "categories": {}, "overall": {}}
    for n in n_src_list:
        results["per_nsrc"][str(n)] = _acc_to_dict(nsrc_acc[n])
        results["categories"][str(n)] = {
            "lang":    {c: _acc_to_dict(cat_acc[n]["lang"][c])    for c in LANG_CATS},
            "gender":  {c: _acc_to_dict(cat_acc[n]["gender"][c])  for c in GENDER_CATS},
            "overlap": {c: _acc_to_dict(cat_acc[n]["overlap"][c]) for c in OVERLAP_CATS},
        }

    def _nm(key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean([_avg(nsrc_acc[n], key) for n in n_src_list]))

    results["overall"] = {
        # Paper metrics: improvement over mixture baseline, vs vocoded target.
        "sdri_vo_avg":         _jv(_nm("sdri_vo")),
        "si_sdri_vo_avg":      _jv(_nm("sisdri_vo")),
        "stoii_avg":           _jv(_nm("stoii")),
        "estoii_avg":          _jv(_nm("estoii")),
        "pesqi_avg":           _jv(_nm("pesqi")),
        # Model uncertainty
        "uq_sigma_avg":        _jv(_nm("uq")),
        "uq_sigma_active_avg": _jv(_nm("uq_active")),
        "uq_sigma_silence_avg":  _jv(_nm("uq_silence")),
        "uq_sigma_target_avg":   _jv(_nm("uq_target")),
        "uq_sigma_overlap_avg":  _jv(_nm("uq_overlap")),
        "uq_uncond_silence_avg": _jv(_nm("uq_uncond_silence")),
        "uq_uncond_target_avg":  _jv(_nm("uq_uncond_target")),
        "uq_uncond_overlap_avg": _jv(_nm("uq_uncond_overlap")),
    }

    # Safe formatter: handles None and NaN without crashing
    ov = results["overall"]
    def _f2(v): return f"{v:.2f}" if v is not None else "N/A"
    def _f4(v): return f"{v:.4f}" if v is not None else "N/A"
    def _f6(v): return f"{v:.6f}" if v is not None else "N/A"
    print(
        f"\n[OVERALL]  "
        f"SDRi(vo)={_f2(ov['sdri_vo_avg'])} dB  "
        f"SI-SDRi(vo)={_f2(ov['si_sdri_vo_avg'])} dB  "
        f"STOIi={_f4(ov['stoii_avg'])}  ESTOIi={_f4(ov['estoii_avg'])}  PESQi={_f4(ov['pesqi_avg'])}  "
        f"UQ(sigma)={_f6(ov['uq_sigma_avg'])}  UQ(sigma_active)={_f6(ov['uq_sigma_active_avg'])}"
    )
    if ov.get("uq_sigma_silence_avg") is not None:
        print(
            f"[STO-sigma OVERALL]  "
            f"S(cond)={_f4(ov['uq_sigma_silence_avg'])}  "
            f"T(cond)={_f4(ov['uq_sigma_target_avg'])}  "
            f"O(cond)={_f4(ov['uq_sigma_overlap_avg'])}  ||  "
            f"S(uncond)={_f4(ov['uq_uncond_silence_avg'])}  "
            f"T(uncond)={_f4(ov['uq_uncond_target_avg'])}  "
            f"O(uncond)={_f4(ov['uq_uncond_overlap_avg'])}"
        )
    results["per_sample"] = per_sample_records
    return results
