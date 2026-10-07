"""Hand-check the variant collapsing: print random groups of listings that were merged."""
import numpy as np
from src.data import load_dataset
from src.diversity import variant_groups
p, _ = load_dataset("data")
g = variant_groups(p)
ids, cnt = np.unique(g, return_counts=True)
multi = ids[cnt > 1]
print(f"{len(multi)} merged groups, {cnt[cnt>1].sum()} listings involved, largest group = {cnt.max()}")
rng = np.random.default_rng(0)
for gid in rng.choice(multi, min(8, len(multi)), replace=False):
    print("\n--- group", gid)
    for t in p.loc[g == gid, "title"].head(4):
        print("   ", t[:110])
