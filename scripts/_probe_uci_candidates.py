#!/usr/bin/env python3
import io, zipfile, urllib.request
import pandas as pd

def get(url, maxb=5_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": "jevtest"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(maxb)

def show(name, df, label=None):
    col = label or df.columns[-1]
    print(f"{name}: shape={df.shape} label={col} counts={df[col].value_counts().to_dict()}")

show("451", pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/451/data.csv"))))
show("529", pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/529/data.csv"))))
show("96", pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/96/data.csv"))), "diagnosis")
show("46", pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/46/data.csv"))), "Class")
show("544", pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/544/data.csv"))))
bm = pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/565/data.csv")))
print("565 cols", [c for c in bm.columns if any(k in c.lower() for k in ("surviv", "status", "death", "dead"))])
show("565", bm, "survival_status" if "survival_status" in bm.columns else bm.columns[-1])
mi = pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/579/data.csv")))
print("579 MI", mi.shape)
for c in mi.columns:
    s = mi[c]
    if s.nunique(dropna=True) == 2 and s.notna().mean() > 0.9 and s.value_counts().min() >= 40:
        print("  bin", c, s.value_counts().to_dict())
raw = get("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")
show("parkinsons", pd.read_csv(io.BytesIO(raw)), "status")
show("244", pd.read_csv(io.BytesIO(get("https://archive.ics.uci.edu/static/public/244/data.csv"))))
z = get("https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip")
with zipfile.ZipFile(io.BytesIO(z)) as zf:
    name = [n for n in zf.namelist() if n.endswith((".data", ".csv", ".dat"))][0]
    df = pd.read_csv(zf.open(name), header=None)
    print("haberman", df.shape, df.iloc[:, -1].value_counts().to_dict())
