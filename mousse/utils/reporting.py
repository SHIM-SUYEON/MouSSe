"""Metric accumulation and summary reporting for evaluation.

Metrics are grouped by language, gender, overlap, and speaker count.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np

# Category definitions 
LANG_CATS = ["EE", "KK", "EK"]
GENDER_CATS = ["MM", "FF", "MF"]
OVERLAP_CATS = ["0", "25", "50", "75", "100"]

# Category accumulator helpers 
def _make_acc():
    return {
        "sdri": 0.0, "n_sdri": 0, # SDRi
        "sisdri": 0.0, "n_sisdri": 0, # SI-SDRi
        "estoii": 0.0, "n_estoii": 0, # ESTOi improvement over mixture baseline
        "uq": 0.0, "n_uq": 0, # UQ sigma_cond
        "n_samples": 0,
    }


def _acc_add(acc, sisdri=None, sdri=None, estoii=None, uq=None):
    if sisdri is not None:
        acc["sisdri"] += sisdri; acc["n_sisdri"] += 1
    if sdri is not None:
        acc["sdri"] += sdri; acc["n_sdri"] += 1
    if estoii is not None:
        acc["estoii"] += estoii; acc["n_estoii"] += 1
    if uq is not None:
        acc["uq"] += uq; acc["n_uq"] += 1


def _merge_acc(dst, src):
    for k in ("sisdri", "sdri", "estoii", "uq"):
        dst[k] += src[k]
    for k in ("n_sisdri", "n_sdri", "n_estoii", "n_uq", "n_samples"):
        dst[k] += src[k]

def _avg(acc, metric):
    s, n = acc[metric], acc[f"n_{metric}"]
    if n > 0:
        return s / n 
    else:
        float("nan")

def _load_meta(mix_path):
    """Load language, gender, overlap, and duration metadata."""
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
    """Return language, gender, and overlap categories."""
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

# Formatting 
def _fmt_metrics(acc):
    dr = _avg(acc, "sdri")      # SDRi
    sd = _avg(acc, "sisdri")    # SI-SDRi
    eti = _avg(acc, "estoii")   # ESTOIi
    uq = _avg(acc, "uq")
    def _f2(v): return f"{v:6.2f}" if not math.isnan(v) else "  N/A "
    def _f4(v): return f"{v:.4f}" if not math.isnan(v) else "  N/A"
    def _f6(v): return f"{v:.6f}" if not math.isnan(v) else "  N/A"
    return (
        f"SDRi={_f2(dr)} dB  SI-SDRi={_f2(sd)} dB  "
        f"ESTOIi={_f4(eti)}  UQ(sigma)={_f6(uq)}"
    )

def _fmt_cell(acc, metric):
    """11-char cell for cross-table."""
    val = _avg(acc, metric)
    if math.isnan(val):
        return f"{'N/A':>11}"
    if metric in ("sdri", "sisdri"):  # dB-valued metrics
        return f"{val:11.2f}"
    return f"{val:11.4f}"

def _print_category_summary(nsrc_acc, cat_acc, n_src_list):
    """Print the category summary."""

    def row(cat_type, value, acc):
        n = acc["n_samples"]
        return (f"  {cat_type:<13}[{value:<14}]: n={n:<6}  {_fmt_metrics(acc)}")

    print("\n" + "=" * 90)
    print("  Per-category summary")
    print("=" * 90)

    # n_src rows
    for n in n_src_list:
        print(row("n_src", str(n), nsrc_acc[n]))

    # lang rows
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

    # overlap rows
    ovlp_agg = {c: _make_acc() for c in OVERLAP_CATS}
    for n in n_src_list:
        for c in OVERLAP_CATS:
            _merge_acc(ovlp_agg[c], cat_acc[n]["overlap"][c])
    for c in ("0", "100", "25", "50", "75"):
        print(row("overlap", f"{c}%", ovlp_agg[c]))

def _print_cross_table(cat_acc, n_src_list):
    """Print one cross table per metric."""
    METRICS = [
        ("sdri", "SDRi (dB)"),
        ("sisdri", "SI-SDRi (dB)"),
        ("estoii", "ESTOIi"),
        # Model uncertainty
        ("uq", "UQ sigma_cond"),
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
    """Print summaries and return the result dictionary."""
    _print_category_summary(nsrc_acc, cat_acc, n_src_list)
    _print_cross_table(cat_acc, n_src_list)

    def _jv(v):
        return None if isinstance(v, float) and math.isnan(v) else round(float(v), 6)

    def _acc_to_dict(acc):
        return {
            "n_samples": acc["n_samples"],
            "sdri": _jv(_avg(acc, "sdri")), "n_sdri": acc["n_sdri"],
            "si_sdri": _jv(_avg(acc, "sisdri")), "n_sisdri": acc["n_sisdri"],
            "estoii": _jv(_avg(acc, "estoii")), "n_estoii": acc["n_estoii"],
            "uq_sigma": _jv(_avg(acc, "uq")), "n_uq": acc["n_uq"],
        }

    results = {"per_nsrc": {}, "categories": {}, "overall": {}}
    for n in n_src_list:
        results["per_nsrc"][str(n)] = _acc_to_dict(nsrc_acc[n])
        results["categories"][str(n)] = {
            "lang": {c: _acc_to_dict(cat_acc[n]["lang"][c]) for c in LANG_CATS},
            "gender": {c: _acc_to_dict(cat_acc[n]["gender"][c]) for c in GENDER_CATS},
            "overlap": {c: _acc_to_dict(cat_acc[n]["overlap"][c]) for c in OVERLAP_CATS},
        }

    def _nm(key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean([_avg(nsrc_acc[n], key) for n in n_src_list]))

    results["overall"] = {
        "sdri_avg": _jv(_nm("sdri")),
        "si_sdri_avg": _jv(_nm("sisdri")),
        "estoii_avg": _jv(_nm("estoii")),
        "uq_sigma_avg": _jv(_nm("uq")),
    }

    ov = results["overall"]
    def _f2(v): return f"{v:.2f}" if v is not None else "N/A"
    def _f4(v): return f"{v:.4f}" if v is not None else "N/A"
    def _f6(v): return f"{v:.6f}" if v is not None else "N/A"
    print(
        f"\n[OVERALL]  "
        f"SDRi={_f2(ov['sdri_avg'])} dB  "
        f"SI-SDRi={_f2(ov['si_sdri_avg'])} dB  "
        f"ESTOIi={_f4(ov['estoii_avg'])}  "
        f"UQ(sigma)={_f6(ov['uq_sigma_avg'])}"
    )
    results["per_sample"] = per_sample_records
    return results
