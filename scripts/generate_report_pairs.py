"""
Synthetic Report-Summary Pair Generator

Generates realistic analytics report → executive summary pairs for SFT.
Each pair includes a detailed multi-section report and a concise 1-page summary.

Author: Fab Admasu
License: MIT
"""

import json
import random
import argparse
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BRANDS = ["Cardivex", "Immunolex", "OncoPrime", "NeuraStar",
          "RespiClear", "DermaShield", "VaxGuard", "EndoBalance"]

CHANNELS = ["TV", "Digital", "Print", "Email", "Rep Visits", "Webinars"]

QUARTERS = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "Q1 2025"]


def generate_report_data():
    """Generate structured analytics data for a brand quarter."""
    brand = random.choice(BRANDS)
    quarter = random.choice(QUARTERS)
    total_budget = round(random.uniform(8, 60), 1)

    channel_data = {}
    weights = np.random.dirichlet(np.ones(len(CHANNELS)) * 2)
    for ch, w in zip(CHANNELS, weights):
        spend = round(float(w * total_budget), 2)
        roi = round(random.uniform(0.5, 7.0), 2)
        roi_ci = (round(max(0, roi - roi * random.uniform(0.2, 0.4)), 2),
                  round(roi + roi * random.uniform(0.2, 0.4), 2))
        channel_data[ch] = {
            "spend_m": spend,
            "roi": roi,
            "roi_ci_lower": roi_ci[0],
            "roi_ci_upper": roi_ci[1],
            "impressions_k": int(spend * random.uniform(50, 200)),
            "response_rate": round(random.uniform(0.01, 0.15), 3),
        }

    market_share = round(random.uniform(5, 35), 1)
    share_change = round(random.uniform(-3, 5), 1)
    trx_volume = int(random.uniform(50000, 500000))

    return {
        "brand": brand,
        "quarter": quarter,
        "total_budget_m": total_budget,
        "channels": channel_data,
        "market_share_pct": market_share,
        "share_change_pp": share_change,
        "trx_volume": trx_volume,
        "top_segments": random.sample(
            ["High-prescribers", "New-to-brand", "Loyalists",
             "Competitive switchers", "Digital-engaged", "Academic KOLs"],
            k=3
        ),
    }


def generate_detailed_report(data: dict) -> str:
    """Generate a multi-section detailed analytics report."""
    d = data
    sorted_ch = sorted(d["channels"].items(), key=lambda x: x[1]["roi"], reverse=True)
    top_ch = sorted_ch[0]
    bottom_ch = sorted_ch[-1]

    report = f"""# {d['brand']} Promotional Effectiveness Report — {d['quarter']}

## 1. Market Overview
{d['brand']} achieved {d['market_share_pct']}% market share in {d['quarter']}, \
a {'+' if d['share_change_pp'] > 0 else ''}{d['share_change_pp']}pp change from prior quarter. \
Total TRx volume was {d['trx_volume']:,}. Total promotional budget was ${d['total_budget_m']}M \
across {len(CHANNELS)} channels.

## 2. Channel Performance Summary
"""
    report += "| Channel | Spend ($M) | ROI | 95% CI | Response Rate |\n"
    report += "|---------|-----------|-----|--------|---------------|\n"
    for ch, cd in sorted_ch:
        report += (f"| {ch} | ${cd['spend_m']}M | {cd['roi']}x | "
                   f"[{cd['roi_ci_lower']}, {cd['roi_ci_upper']}] | "
                   f"{cd['response_rate']*100:.1f}% |\n")

    report += f"""
## 3. Key Findings
1. **Highest ROI:** {top_ch[0]} delivers {top_ch[1]['roi']}x return \
(95% CI: [{top_ch[1]['roi_ci_lower']}, {top_ch[1]['roi_ci_upper']}])
2. **Lowest ROI:** {bottom_ch[0]} at {bottom_ch[1]['roi']}x — consider reducing allocation
3. **Top Segments:** {', '.join(d['top_segments'])}
4. **Market Share:** {'Gained' if d['share_change_pp'] > 0 else 'Lost'} \
{abs(d['share_change_pp'])}pp market share

## 4. Statistical Notes
- ROI estimates derived from Bayesian Marketing Mix Model (PyMC-Marketing)
- Adstock: geometric decay; Saturation: Hill function
- All confidence intervals are 95% Bayesian credible intervals
- Results should be interpreted with standard MMM caveats (correlation ≠ causation)
"""
    return report


def generate_executive_summary(data: dict) -> str:
    """Generate a concise executive summary from the data."""
    d = data
    sorted_ch = sorted(d["channels"].items(), key=lambda x: x[1]["roi"], reverse=True)
    top_ch = sorted_ch[0]
    bottom_ch = sorted_ch[-1]

    total_roi = sum(cd["spend_m"] * cd["roi"] for _, cd in sorted_ch) / d["total_budget_m"]

    summary = f"""## Executive Summary: {d['brand']} — {d['quarter']}

**Market Position:** {d['market_share_pct']}% share \
({'+' if d['share_change_pp'] > 0 else ''}{d['share_change_pp']}pp QoQ) | \
{d['trx_volume']:,} TRx | ${d['total_budget_m']}M promotional budget

**Key Findings:**
1. {top_ch[0]} is the highest-ROI channel at {top_ch[1]['roi']}x \
(95% CI: [{top_ch[1]['roi_ci_lower']}, {top_ch[1]['roi_ci_upper']}]) — \
recommend increasing investment
2. {bottom_ch[0]} underperforms at {bottom_ch[1]['roi']}x — \
recommend reducing by ~${bottom_ch[1]['spend_m'] * 0.3:.1f}M
3. Portfolio-weighted ROI: {total_roi:.1f}x across all channels

**Recommendations:**
- Shift ${bottom_ch[1]['spend_m'] * 0.3:.1f}M from {bottom_ch[0]} → {top_ch[0]}
- Priority segments: {', '.join(d['top_segments'][:2])}
- Monitor {top_ch[0]} saturation (current spend may approach diminishing returns)

**Confidence:** Bayesian credible intervals provided; statistical significance confirmed for top 3 channels."""
    return summary


def generate_dataset(n_reports: int, output_dir: Path):
    """Generate report-summary training pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for _ in range(n_reports):
        data = generate_report_data()
        report = generate_detailed_report(data)
        summary = generate_executive_summary(data)

        examples.append({
            "instruction": f"Generate a concise executive summary from this analytics report:\n\n{report}",
            "response": summary,
            "metadata": {"brand": data["brand"], "quarter": data["quarter"]},
        })

    random.shuffle(examples)
    split = int(len(examples) * 0.9)

    for split_name, split_data in [("train", examples[:split]), ("eval", examples[split:])]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")

    print(f"✅ Generated {len(examples[:split])} train + {len(examples[split:])} eval pairs")
    print(f"   📁 {output_dir}/train.jsonl, eval.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reports", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="data")
    main_args = parser.parse_args()
    generate_dataset(main_args.n_reports, Path(main_args.output_dir))


if __name__ == "__main__":
    main()
