import pandas as pd, numpy as np
df = pd.read_csv("projects/XCEPT_Article_1/outputs/intra_org_esmo/esmo_cache_intra.csv")
print("Cache rows:", len(df))
for org in df["org"].unique():
    for combo in ["pc", "cp"]:
        sub = df[(df["org"]==org) & (df["combo"]==combo)]
        n_vals = sub["n_esmos"].values
        pr = float(np.mean(n_vals > 0))
        mn = float(np.mean(n_vals))
        sd = float(np.std(n_vals))
        nsigs = sub["sig_hash"].nunique()
        print(f"{org} {combo}: pairs={len(n_vals)}  Pr={pr:.3f}  mean_N={mn:.1f}  std={sd:.1f}  distinct_sigs={nsigs}")
