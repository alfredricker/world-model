"""print a run of card 152 compactly: show.py runs/152/<name>_<by>.json"""
import sys, json
for d in json.load(open(sys.argv[1])):
    print("world", d["world"], d["by"], d["kind_of"], "| episodes", d["train_episodes"], "| stored", d["individuals_stored"], "identity mistakes", d["identity_mistakes_in_training"],
          "| codes", d["shape_codes"], d["colour_codes"], "| kinds", d["kinds"], "purity", d["kind_purity"], d["kinds_by_true_behaviour"], "| looks", d["looks_share"], "|", d["seconds"], "s")
    print("  rounds (kinds, moved, unexplained, merged)", [(r["kinds"], r["moved"], r["unexplained"], r.get("merged", False)) for r in d["kind_rounds"]])
    r = d["recognition"]
    print("  recognition", r["views"], {k: (v["n"], v["right"], v["right_known"], v["right_new"], "not settled", v["where_not_settled"], "no frame", v["no_object_frame_known"], "place", v["place_right_known"]) for k, v in r.items() if isinstance(v, dict)}, r["energy_of_right_known"], r["energy_of_new"])
    for nm in ("new_objects", "known_objects"):
        t = d[nm]; print(" ", nm, t["informative_object_steps"], "recognised", t["recognised_right"], "kind from looks", t["kind_from_looks_right"])
        for k, v in t["rows"].items():
            print("      effects |", k, v)
        for k, v in t["take"].items():
            print("      take    |", k, v)
