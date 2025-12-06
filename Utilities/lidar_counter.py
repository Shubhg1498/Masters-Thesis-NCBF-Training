# split_csv.py
import glob, pandas as pd, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--pattern', default='data/run*_rot.csv',
                    help='Glob for the CSV files to merge')
parser.add_argument('--train_ratio', type=float, default=0.70)
parser.add_argument('--val_ratio',   type=float, default=0.15)
args = parser.parse_args()

# 1) Load and concatenate
files = sorted(glob.glob(args.pattern))
if not files:
    raise FileNotFoundError(f"No CSV files matched {args.pattern}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
n  = len(df)
train_end = int(args.train_ratio * n)
val_end   = int((args.train_ratio + args.val_ratio) * n)

# 2) Save splits
os.makedirs('data', exist_ok=True)
df.iloc[:train_end]        .to_csv('data/train.csv', index=False)
df.iloc[train_end:val_end] .to_csv('data/val.csv',   index=False)
df.iloc[val_end:]          .to_csv('data/test.csv',  index=False)

print(f"Rows  train/val/test: {len(df.iloc[:train_end])}/"
      f"{len(df.iloc[train_end:val_end])}/"
      f"{len(df.iloc[val_end:])}")
print("Output written to data/train.csv  data/val.csv  data/test.csv")
