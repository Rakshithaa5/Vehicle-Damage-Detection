import pandas as pd
from pathlib import Path

# Find results.csv
csvs = list(Path('runs').rglob('results.csv'))
if not csvs:
    print("No results.csv found!")
else:
    for p in csvs:
        print(f"\nFound: {p}")
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        print("\nAll columns:", list(df.columns))
        last = df.iloc[-1]
        print(f"\n{'='*45}")
        print("  YOLOv8m FINAL RESULTS (epoch 100)")
        print(f"{'='*45}")
        # Print all metrics
        for col in df.columns:
            if any(k in col for k in ['mAP','precision','recall','loss']):
                try:
                    print(f"  {col.strip():<30}: {float(last[col]):.4f}")
                except:
                    pass
        print(f"{'='*45}")
