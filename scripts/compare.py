"""Collect results/*.json into results/RESULTS.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json"))]
rows.sort(key=lambda r: -r["overall"]["f1"])

out = ["# Test-set results", "",
       f"Held-out test split: {rows[0]['overall']['n']} images (pothole + normal), never seen in training. Threshold 0.5.", "",
       "| Model | Accuracy | Precision | Recall | F1 | AUC | p50 latency | Device | Notes |",
       "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    o, l = r["overall"], r["latency_ms"]
    out.append(f"| {r['name']} | {o['accuracy']:.1%} | {o['precision']:.1%} | {o['recall']:.1%} | {o['f1']:.3f} | "
               f"{o['auc']:.3f} | {l['p50']:.1f} ms | {r['device']} | {r['notes']} |")
out += ["", "## F1 by data source", "", "| Model | " + " | ".join(rows[0]["by_source"]) + " |",
        "|---|" + "---|" * len(rows[0]["by_source"])]
for r in rows:
    out.append(f"| {r['name']} | " + " | ".join(f"{s['f1']:.3f} (n={s['n']})" for s in r["by_source"].values()) + " |")
(ROOT / "results/RESULTS.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
