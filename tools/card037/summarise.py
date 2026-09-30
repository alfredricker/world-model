"""Card 037: results.json from the run files (arms 1-4, the gate).

Run: python tools/card037/summarise.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
COLOURS = ("yellow", "purple")
WORLDS = ("key", "switch", "either", "both")
CRIT = ("pick up the {c} key", "drop the {c} key", "forward onto the open {c} door")


def load(*names):
    seeds = []
    for n in names:
        p = RUNS / n
        if p.exists():
            d = json.loads(p.read_text())
            seeds += next(iter(d["arms"].values()))["seeds"]
    return sorted(seeds, key=lambda s: s["seed"])


def arm_summary(seeds):
    out = {"seeds_run": [s["seed"] for s in seeds],
           "criteria_seeds": {c: sum(bool(s["verdicts"][c]) for s in seeds) for c in ("criterion_1", "criterion_2", "criterion_3")},
           "model_too_big": [s["seed"] for s in seeds if "error" in s]}
    ok = [s for s in seeds if "error" not in s]
    out["criterion_1_failing_seeds"] = [s["seed"] for s in ok if not s["verdicts"]["criterion_1"]]
    for col in COLOURS:
        cases, acting = {}, []
        for s in ok:
            q = s["worlds"]["key"][f"test_a_{col}"]
            for k, v in q["cases"].items():
                if "reported" in k:
                    continue
                cases[k] = cases.get(k, 0) + int((v["exact_share"] or 0) >= 0.99)
            acting.append([s["seed"], q["acting"]["success"], q["acting"]["steps_ratio_to_shortest"]])
        out[col] = {"cases_at_99pct_seeds": cases, "acting_success_and_steps_ratio": acting,
                    "world_b_acting": [s["worlds"]["switch"][f"test_b_{col}"]["acting"]["success"] for s in ok]}
        if ok and "new_tile_votes" in ok[0]:
            out[col]["forward_onto_open_door"] = [
                {"seed": s["seed"],
                 "draw_vote": (s["new_tile_votes"]["draw"].get(f"open door {col}") or {}).get("vote"),
                 "exact_share": s["worlds"]["key"][f"test_a_{col}"]["cases"][f"forward onto the open {col} door"]["exact_share"]}
                for s in ok]
        if ok and "recall" in ok[0]:
            out[col]["recall_own_right_seeds"] = {
                case: sum(bool(s["recall"][col]["predicted"][case]["right"]) for s in ok)
                for case in ok[0]["recall"][col]["predicted"]}
    if ok and "recall_reports" in ok[0]:
        worse = [[s["seed"], w, kind] for s in ok for w, r in zip(WORLDS, s["recall_reports"]) for kind, v in r.items()
                 if v["loo_start"] is not None and v["outcomes"] > 1 and not v["loo_fitted"] > v["loo_start"]]
        out["lambda_fit_not_better"] = worse
        out["tuples_per_world"] = {s["seed"]: [s["worlds"][w]["entries"].get("tuples") for w in WORLDS] for s in ok}
        out["model_seconds"] = {s["seed"]: s["model_seconds"] for s in ok}
    return out


def main():
    gate = json.loads((RUNS / "037_gate.json").read_text())
    a1 = json.loads((RUNS / "037_arm1.json").read_text())["arms"]["1_labels"]
    res = {"card": "037", "seeds": list(range(400, 410)),
           "gate": {"alpha": gate["alpha"], "loo_right_of_800": gate["loo_right"],
                    "seeds_named_apart": gate["seeds_named_apart"], "names_pass": gate["names_pass"]},
           "arm_1_labels": a1["verdicts"],
           "arms": {"2_recall": arm_summary(load("037_arm2_a.json", "037_arm2_b.json")),
                    "3_counted": arm_summary(load("037_arm3.json")),
                    "4_recall_lambda_at_start": arm_summary(load("037_arm4_a.json", "037_arm4_b.json"))},
           "first_launch_of_arm_2": {s["seed"]: s["verdicts"] for s in load("037_arm2_first_try.json")}}
    res["arms"]["4_recall_lambda_at_start"]["stopped_at_25_minutes"] = {
        "not_finished": {"400": "first world's model not built after 25 minutes", "409": "building for about 10 minutes"},
        "not_run": [401, 402, 403, 404]}
    out = ROOT / "experiments" / "037-recall-in-the-planner" / "results.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v["criteria_seeds"] for k, v in res["arms"].items()}))


if __name__ == "__main__":
    main()
