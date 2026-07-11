#!/usr/bin/env python3
"""Compute PCC/SRCC between rFID/GAR-FID and gFID.

The script reads GAR-FID values from results.csv and keeps the rFID/gFID
numbers from the paper tables in this file so the correlation table can be
recomputed reproducibly.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError as exc:
    raise SystemExit("scipy is required: python3 -m pip install scipy") from exc


ETAS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")

TOKENIZERS = {
    "b": (
        "sdvae_b",
        "invae_b",
        "fluxvae_b",
        "qwvae_b",
        "sd3vae_b",
        "eqvae_b",
        "vavae_b",
        "vavae64_b",
        "softvq_b",
        "maetok_b",
        "detok_b",
        "dmvae_b",
        "repaevae_b",
    ),
    "xl": (
        "sdvae_xl",
        "invae_xl",
        "fluxvae_xl",
        "qwvae_xl",
        "sd3vae_xl",
        "eqvae_xl",
        "vavae_xl",
        "vavae64_xl",
        "softvq_xl",
        "maetok_xl",
        "detok_xl",
        "dmvae_xl",
        "repaevae_xl",
        "rae_xl",
    ),
}

RFID = {
    "sdvae": 0.74,
    "invae": 0.26,
    "fluxvae": 0.16,
    "qwvae": 1.52,
    "sd3vae": 0.21,
    "eqvae": 0.59,
    "vavae": 0.30,
    "vavae64": 0.15,
    "softvq": 0.59,
    "maetok": 0.61,
    "detok": 0.59,
    "dmvae": 0.72,
    "repaevae": 0.56,
    "rae": 0.63,
}

GFID = {
    "b": {
        "sdvae_b": {"nocfg": 46.64, "cfg": 9.91},
        "invae_b": {"nocfg": 48.28, "cfg": 10.87},
        "fluxvae_b": {"nocfg": 62.92, "cfg": 14.38},
        "qwvae_b": {"nocfg": 48.03, "cfg": 10.97},
        "sd3vae_b": {"nocfg": 51.58, "cfg": 11.89},
        "eqvae_b": {"nocfg": 37.49, "cfg": 9.32},
        "vavae_b": {"nocfg": 18.88, "cfg": 6.01},
        "vavae64_b": {"nocfg": 33.56, "cfg": 7.76},
        "softvq_b": {"nocfg": 29.48, "cfg": 7.48},
        "maetok_b": {"nocfg": 13.75, "cfg": 5.69},
        "detok_b": {"nocfg": 20.17, "cfg": 6.95},
        "dmvae_b": {"nocfg": 8.38, "cfg": 4.89},
        "repaevae_b": {"nocfg": 25.75, "cfg": 6.46},
    },
    "xl": {
        "sdvae_xl": {"nocfg": 25.91, "cfg": 6.33},
        "invae_xl": {"nocfg": 25.65, "cfg": 6.56},
        "fluxvae_xl": {"nocfg": 34.06, "cfg": 7.82},
        "qwvae_xl": {"nocfg": 23.62, "cfg": 6.19},
        "sd3vae_xl": {"nocfg": 26.38, "cfg": 6.31},
        "eqvae_xl": {"nocfg": 20.81, "cfg": 6.24},
        "vavae_xl": {"nocfg": 8.57, "cfg": 4.20},
        "vavae64_xl": {"nocfg": 15.40, "cfg": 5.36},
        "softvq_xl": {"nocfg": 15.88, "cfg": 5.14},
        "maetok_xl": {"nocfg": 6.27, "cfg": 3.74},
        "detok_xl": {"nocfg": 11.97, "cfg": 4.83},
        "dmvae_xl": {"nocfg": 4.65, "cfg": 3.39},
        "repaevae_xl": {"nocfg": 12.95, "cfg": 4.17},
        "rae_xl": {"nocfg": 4.25, "cfg": 3.50},
    },
}

IFID_DAGGER = {
    ("b", "nocfg"): (0.85, 0.86),
    ("b", "cfg"): (0.82, 0.84),
    ("xl", "nocfg"): (0.89, 0.91),
    ("xl", "cfg"): (0.88, 0.92),
}

IFID_STAR = {
    "sdvae": 59.8769,
    "invae": 41.5253,
    "fluxvae": 66.2624,
    "qwvae": 30.4405,
    "sd3vae": 36.7194,
    "eqvae": 47.7489,
    "vavae": 20.1396,
    "vavae64": 21.6550,
    "softvq": 26.8990,
    "maetok": 14.3977,
    "detok": 17.8115,
    "dmvae": 8.5465,
    "repaevae": 37.0264,
    # RAE failed because the DINOv2 encoder was not available in the offline HF cache.
}


def tokenizer_base(tokenizer: str) -> str:
    if tokenizer.endswith("_b"):
        return tokenizer[:-2]
    if tokenizer.endswith("_xl"):
        return tokenizer[:-3]
    return tokenizer


def corr_pair(metric_values: list[float], gfid_values: list[float]) -> tuple[float, float]:
    if len(metric_values) != len(gfid_values) or len(metric_values) < 2:
        return math.nan, math.nan
    pcc, _ = pearsonr(metric_values, gfid_values)
    srcc, _ = spearmanr(metric_values, gfid_values)
    return float(pcc), float(srcc)


def read_garfid(path: Path) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            group_id = row["group_id"].replace("_0421", "")
            key = (row["model"], group_id, row["eta"])
            value = float(row["garfid"])
            old = values.get(key)
            if old is not None and round(old, 8) != round(value, 8):
                raise ValueError(f"Conflicting GAR-FID for {key}: {old} vs {value}")
            values[key] = value
    return values


def collect_points(
    garfid: dict[tuple[str, str, str], float],
    model: str,
    target: str,
    metric: str,
    eta: str | None,
    exclude_rae: bool,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for tokenizer in TOKENIZERS[model]:
        if exclude_rae and tokenizer_base(tokenizer) == "rae":
            continue
        y = GFID[model][tokenizer][target]
        if y is None:
            continue
        if metric == "rfid":
            x = RFID[tokenizer_base(tokenizer)]
        elif metric == "ifid":
            base = tokenizer_base(tokenizer)
            if base not in IFID_STAR:
                continue
            x = IFID_STAR[base]
        elif metric == "garfid":
            if eta is None:
                raise ValueError("eta is required for garfid")
            key = (model, tokenizer, eta)
            if key not in garfid:
                continue
            x = garfid[key]
        else:
            raise ValueError(metric)
        xs.append(x)
        ys.append(y)
    return xs, ys


def collect_mixed_points(
    garfid: dict[tuple[str, str, str], float],
    target: str,
    metric: str,
    eta: str | None,
    exclude_rae: bool,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for model in ("b", "xl"):
        x_part, y_part = collect_points(garfid, model, target, metric, eta, exclude_rae)
        xs.extend(x_part)
        ys.extend(y_part)
    return xs, ys


def format_value(value: float, best: bool = False) -> str:
    text = f"{value:.2f}"
    return f"\\textbf{{{text}}}" if best else text


def build_rows(garfid: dict[tuple[str, str, str], float], exclude_rae: bool) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    metrics = [("rFID$^{\\star}$", "rfid", None), ("iFID$^{\\star}$", "ifid", None)]
    metrics += [(f"GAR-FID ($\\eta_t={eta}$)", "garfid", eta) for eta in ETAS]

    for label, metric, eta in metrics:
        row: dict[str, str | float] = {"metric": label}
        for group in ("b", "xl", "mixed"):
            for target in ("nocfg", "cfg"):
                if group == "mixed":
                    xs, ys = collect_mixed_points(garfid, target, metric, eta, exclude_rae)
                else:
                    xs, ys = collect_points(garfid, group, target, metric, eta, exclude_rae)
                pcc, srcc = corr_pair(xs, ys)
                row[f"{group}_{target}_pcc"] = pcc
                row[f"{group}_{target}_srcc"] = srcc
                row[f"{group}_{target}_n"] = len(xs)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str | float]], path: Path) -> None:
    fields = ["metric"]
    for group in ("b", "xl", "mixed"):
        for target in ("nocfg", "cfg"):
            fields.extend([f"{group}_{target}_pcc", f"{group}_{target}_srcc", f"{group}_{target}_n"])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows: list[dict[str, str | float]], path: Path, include_ifid: bool) -> None:
    value_cols = [
        "b_nocfg_pcc",
        "b_nocfg_srcc",
        "b_cfg_pcc",
        "b_cfg_srcc",
        "xl_nocfg_pcc",
        "xl_nocfg_srcc",
        "xl_cfg_pcc",
        "xl_cfg_srcc",
        "mixed_nocfg_pcc",
        "mixed_nocfg_srcc",
        "mixed_cfg_pcc",
        "mixed_cfg_srcc",
    ]

    # Bold maxima per column among computed rows only.
    best: dict[str, float] = {}
    for col in value_cols:
        vals = [float(row[col]) for row in rows if not math.isnan(float(row[col]))]
        best[col] = max(vals) if vals else math.nan

    lines: list[str] = []
    if include_ifid:
        lines.append(
            "iFID$^{\\dagger}$ & 0.85 & 0.86 & 0.82 & 0.84 & "
            "0.89 & 0.91 & 0.88 & 0.92 & -- & -- & -- & -- \\\\"
        )
    for row in rows:
        values = []
        for col in value_cols:
            value = float(row[col])
            if math.isnan(value):
                values.append("--")
            else:
                values.append(format_value(value, abs(value - best[col]) < 0.005))
        lines.append(f"{row['metric']} & " + " & ".join(values) + " \\\\")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-csv", type=Path, default=Path("results.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("correlation_results.csv"))
    parser.add_argument("--out-tex", type=Path, default=Path("correlation_latex_rows.tex"))
    parser.add_argument(
        "--exclude-rae",
        action="store_true",
        help="Exclude RAE from XL/mixed correlations even when an RAE gFID value is available.",
    )
    parser.add_argument("--no-ifid", action="store_true", help="Do not prepend iFID dagger row in LaTeX output.")
    args = parser.parse_args()

    garfid = read_garfid(args.results_csv)
    rows = build_rows(garfid, exclude_rae=args.exclude_rae)
    write_csv(rows, args.out_csv)
    write_latex(rows, args.out_tex, include_ifid=not args.no_ifid)

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_tex}")
    print("n columns in CSV include sample counts used for each correlation.")


if __name__ == "__main__":
    main()
