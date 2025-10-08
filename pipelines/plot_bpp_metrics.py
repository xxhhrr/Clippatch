import argparse, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="ssim")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(f"列 {args.metric} 不在CSV列：{list(df.columns)}")
    dd = df.dropna(subset=["bpp", args.metric])

    plt.figure(figsize=(6,4))
    plt.scatter(dd["bpp"], dd[args.metric], s=8)
    plt.xlabel("bpp"); plt.ylabel(args.metric.upper())
    plt.title(f"{args.metric.upper()} vs. bpp (n={len(dd)})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    out = args.out or f"bpp_vs_{args.metric}.png"
    plt.savefig(out, dpi=200)
    print("saved ->", out)

if __name__ == "__main__":
    main()

#  python tools/plot_bpp_metric.py --csv outputs/metrics_bpp_ssim_psnr.csv --metric ssim --out outputs/bpp_vs_ssim.png
