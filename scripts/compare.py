"""Collect results/<model>.json (test split) into results/RESULTS.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# <model>.json only: skip <model>.val.json, .thresholds.json, .robustness.json
rows = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("*.json")) if p.name.count(".") == 1]
rows.sort(key=lambda r: -(r["overall"]["auc"] or 0))
sources = sorted({s for r in rows for s in r["by_source"]})
captures = sorted({c for r in rows for c in r.get("by_capture", {})})


def cell(groups, key):
    s = groups.get(key)
    return f"{s['auc']:.3f} / {s['accuracy']:.1%}" if s and s["auc"] is not None else "-"


out = ["# Test-set results", "",
       f"Held-out test split: {rows[0]['overall']['n']} images (pothole + normal), never seen in training. "
       "Cutoff 0.5 for every model here; per-camera cutoffs for the default model are in README.", "",
       "Sorted by AUC, which compares models without picking a cutoff (0.5 = coin flip, 1.0 = perfect).", "",
       "| Model | AUC | Accuracy | Precision | Recall | F1 | p50 latency | Device | Notes |",
       "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    o, l = r["overall"], r["latency_ms"]
    out.append(f"| {r['name']} | {o['auc']:.3f} | {o['accuracy']:.1%} | {o['precision']:.1%} | {o['recall']:.1%} | "
               f"{o['f1']:.3f} | {l['p50']:.1f} ms | {r['device']} | {r['notes']} |")
out += ["", "## By camera type (AUC / accuracy)", "", "| Model | " + " | ".join(captures) + " |",
        "|---|" + "---|" * len(captures)]
out += [f"| {r['name']} | " + " | ".join(cell(r.get("by_capture", {}), c) for c in captures) + " |" for r in rows]
out += ["", "## By data source (AUC / accuracy)", "", "| Model | " + " | ".join(sources) + " |",
        "|---|" + "---|" * len(sources)]
out += [f"| {r['name']} | " + " | ".join(cell(r["by_source"], s) for s in sources) + " |" for r in rows]
(ROOT / "results/RESULTS.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
