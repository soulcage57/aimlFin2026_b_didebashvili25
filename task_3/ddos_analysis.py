"""
DDoS Attack Detection via Regression Analysis
==============================================
Input  : Web server access log (Apache/Nginx Combined Log Format)
Method : Polynomial regression baseline + residual anomaly detection
Output : DDoS attack time intervals + visualizations

Usage:
    python ddos_analysis.py --log log.txt
"""

import re
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. PARSE LOG FILE
# ─────────────────────────────────────────────────────────────
def parse_log(filepath):
    """Parse web server log and aggregate request counts per minute."""
    pattern = r'(\d+\.\d+\.\d+\.\d+).*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    per_minute      = defaultdict(int)
    per_minute_ips  = defaultdict(set)

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                ip  = m.group(1)
                ts  = datetime.strptime(m.group(2), '%Y-%m-%d %H:%M:%S')
                key = ts.strftime('%H:%M')
                per_minute[key]     += 1
                per_minute_ips[key].add(ip)

    sorted_minutes = sorted(per_minute.items())
    labels      = [k for k, _ in sorted_minutes]
    counts      = np.array([v for _, v in sorted_minutes])
    unique_ips  = np.array([len(per_minute_ips[k]) for k in labels])
    return labels, counts, unique_ips


# ─────────────────────────────────────────────────────────────
# 2. POLYNOMIAL REGRESSION BASELINE
# ─────────────────────────────────────────────────────────────
def fit_baseline(counts, degree=15):
    """
    Fit a polynomial regression to the request-rate time series.
    The fitted curve represents the expected 'normal' traffic baseline.
    Residuals (observed - fitted) expose anomalous spikes.
    """
    x = np.arange(len(counts)).reshape(-1, 1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x, counts)
    baseline = model.predict(x)
    residuals = counts - baseline
    r2 = r2_score(counts, baseline)
    return baseline, residuals, r2


# ─────────────────────────────────────────────────────────────
# 3. ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────
def detect_anomalies(residuals, sigma_threshold=2.5):
    """
    Flag minutes whose residual exceeds mean + sigma_threshold * std.
    Groups consecutive (or near-consecutive) anomalous minutes into
    attack intervals.
    """
    mu    = np.mean(residuals)
    sigma = np.std(residuals)
    threshold    = mu + sigma_threshold * sigma
    anomaly_mask = residuals > threshold
    anomaly_idx  = np.where(anomaly_mask)[0]

    # Group consecutive anomaly indices (gap <= 2 minutes)
    groups = []
    if len(anomaly_idx) > 0:
        start = prev = anomaly_idx[0]
        for idx in anomaly_idx[1:]:
            if idx - prev <= 2:
                prev = idx
            else:
                groups.append((start, prev))
                start = prev = idx
        groups.append((start, prev))

    return anomaly_mask, groups, threshold


# ─────────────────────────────────────────────────────────────
# 4. VISUALISATION  (3-panel regression report)
# ─────────────────────────────────────────────────────────────
BG = "#0d1117"; PANEL = "#161b22"; TEXT = "#c9d1d9"
ACCENT = "#58a6ff"; GREEN = "#3fb950"; RED = "#f85149"; YELLOW = "#e3b341"


def style_ax(ax, title, xl="Time (HH:MM)", yl="Requests / min"):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.set_title(title, color=TEXT, fontsize=12, pad=8, fontweight="bold")
    ax.set_xlabel(xl, color=TEXT, fontsize=10)
    ax.set_ylabel(yl, color=TEXT, fontsize=10)
    ax.grid(color="#21262d", lw=0.5, alpha=0.6)


def plot_regression_report(labels, counts, baseline, residuals,
                            anomaly_mask, groups, threshold, r2,
                            out_path="ddos_analysis.png"):
    x = np.arange(len(counts))
    tick_step = max(1, len(labels) // 20)
    tick_pos  = x[::tick_step]
    tick_lbl  = [labels[i] for i in tick_pos]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), facecolor=BG)

    # ── Panel 1: Raw time-series with attack shading ──
    ax = axes[0]
    style_ax(ax, "Web Server Request Rate — Full Window")
    ax.bar(x, counts, color=ACCENT, alpha=0.6, width=0.8, label="Requests/min")
    for s, e in groups:
        ax.axvspan(s - 0.5, e + 0.5, color=RED, alpha=0.25, zorder=0)
        ax.annotate(
            f"DDoS\n{labels[s]}–{labels[e]}",
            xy=((s + e) / 2, counts[s:e+1].max()),
            xytext=((s + e) / 2, counts[s:e+1].max() + 250),
            ha='center', color=RED, fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.2)
        )
    ax.axhline(threshold, color=YELLOW, lw=1.5, linestyle='--',
               label=f'Detection threshold ({threshold:.0f} req/min)')
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha='right')
    ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=9)

    # ── Panel 2: Polynomial regression baseline ──
    ax = axes[1]
    style_ax(ax, f"Polynomial Regression Baseline (degree=15, R²={r2:.3f})")
    ax.plot(x, counts, color=ACCENT, lw=1.2, alpha=0.7, label="Observed req/min")
    ax.plot(x, baseline, color=GREEN, lw=2.2, label=f"Poly fit (R²={r2:.3f})")
    ax.scatter(x[anomaly_mask], counts[anomaly_mask],
               color=RED, s=70, zorder=5, label="Detected anomalies")
    for s, e in groups:
        ax.axvspan(s - 0.5, e + 0.5, color=RED, alpha=0.18, zorder=0)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha='right')
    ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=9)

    # ── Panel 3: Residuals ──
    ax = axes[2]
    style_ax(ax, "Regression Residuals (Observed − Baseline)",
             yl="Residual (req/min)")
    ax.bar(x, residuals, color=ACCENT, alpha=0.55, width=0.8, label="Residual")
    ax.bar(x[anomaly_mask], residuals[anomaly_mask],
           color=RED, alpha=0.85, width=0.8, label="Anomaly residual")
    ax.axhline(threshold - np.mean(residuals), color=YELLOW, lw=1.5,
               linestyle='--', label=f'+2.5σ threshold')
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)
    for s, e in groups:
        ax.axvspan(s - 0.5, e + 0.5, color=RED, alpha=0.18, zorder=0)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha='right')
    ax.legend(facecolor=PANEL, edgecolor="#30363d", labelcolor=TEXT, fontsize=9)

    fig.suptitle(
        "DDoS Attack Detection via Polynomial Regression — Web Server Log",
        color=TEXT, fontsize=14, fontweight='bold', y=1.005
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DDoS detection via regression")
    parser.add_argument("--log", default="log.txt", help="Path to log file")
    parser.add_argument("--degree", type=int, default=15,
                        help="Polynomial degree for baseline (default: 15)")
    parser.add_argument("--sigma", type=float, default=2.5,
                        help="Sigma threshold for anomaly (default: 2.5)")
    args = parser.parse_args()

    print(f"[1/4] Parsing log file: {args.log}")
    labels, counts, unique_ips = parse_log(args.log)
    print(f"      {len(counts)} minute-buckets, {counts.sum()} total requests")
    print(f"      Time range: {labels[0]} → {labels[-1]}")

    print(f"\n[2/4] Fitting polynomial regression (degree={args.degree})...")
    baseline, residuals, r2 = fit_baseline(counts, degree=args.degree)
    print(f"      R² = {r2:.4f}")

    print(f"\n[3/4] Detecting anomalies (σ threshold = {args.sigma})...")
    anomaly_mask, groups, threshold = detect_anomalies(residuals, args.sigma)

    if not groups:
        print("      No DDoS intervals detected.")
    else:
        print(f"      Found {len(groups)} attack interval(s):")
        for s, e in groups:
            total_reqs = counts[s:e+1].sum()
            peak       = counts[s:e+1].max()
            normal_avg = np.mean(counts[~anomaly_mask])
            ratio      = peak / normal_avg
            print(f"\n      ┌─ DDoS Attack Interval ──────────────────────")
            print(f"      │  Start  : {labels[s]}")
            print(f"      │  End    : {labels[e]}")
            print(f"      │  Total  : {total_reqs:,} requests")
            print(f"      │  Peak   : {peak:,} req/min  ({ratio:.1f}× normal avg)")
            print(f"      └────────────────────────────────────────────")

    print(f"\n[4/4] Generating visualizations...")
    plot_regression_report(labels, counts, baseline, residuals,
                            anomaly_mask, groups, threshold, r2)
    print("\nDone.")


if __name__ == "__main__":
    main()
