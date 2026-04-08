import csv
import json
from pathlib import Path

csv_path = Path(r"d:/Documents/LMIS/Test_text_for_Covid19.csv")
out_path = Path(__file__).parent / 'test_prompts.json'

mapping = {}
with csv_path.open('r', encoding='utf-8') as f:
    # Semicolon separated, skip header
    for i, line in enumerate(f):
        if i == 0:
            continue
        parts = line.strip().split(';')
        if len(parts) < 2:
            continue
        imgname = parts[0].strip()
        text = parts[1].strip()
        if not imgname:
            continue
        # remove leading 'mask_' if present
        if imgname.startswith('mask_'):
            imgname = imgname[len('mask_'):]
        mapping[imgname] = text

with out_path.open('w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(mapping)} entries to {out_path}")
