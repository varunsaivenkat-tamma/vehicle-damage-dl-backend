import gzip
import shutil

src = "cost_model.pkl"
dst = "cost_model.pkl.gz"

with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)

print("Created:", dst)
