"""compact view of runs/155 files: show.py <files...>"""
import sys, json
for f in sys.argv[1:]:
    for r in json.load(open(f)):
        print("world %d by %s, %ds: %d patterns, %d objects with transitions %s of true %s; transitions %d of %d; classes %d %s purity %s by true behaviour %s" % (r["world"], r["by"], r["seconds"], r["patterns"], r["objects_with_transitions"],
              r["objects_by_true_behaviour"], r["true_objects_in_training_by_behaviour"], r["informative_used"], r["informative_with_masks"], r["classes"], r["class_sizes"], r["class_purity"], r["classes_by_true_behaviour"]))
        for part in ("known_objects", "new_objects"):
            a = r[part]; A = "appearance and own transitions"
            print("   %-14s bound %s %s | effects all %s (bound %s, one %s) bound-only %s (%s) | take all %s (bound %s) takeable %s sticky %s fixed %s exceptions %s bound-only %s" % (part, a["objects_bound"], {k: v["bound"] for k, v in a["bound_by_behaviour"].items()},
                  a["rows"]["all"][A], a["rows"]["all"]["true behaviour given"], a["rows"]["all"]["one class for all"], a["rows"].get("bound", {}).get(A), a["rows"].get("bound", {}).get("true behaviour given"),
                  a["take"]["all"][A], a["take"]["all"]["true behaviour given"], a["take"]["takeable"][A], a["take"]["sticky"][A], a["take"]["fixed"][A], a["take"].get("exceptions", {}).get(A), a["take"].get("bound", {}).get(A)))
        a = r["new_objects_mask_at_test"]; A = "looks only"
        print("   new, mask at test: effects all %s takeable %s (%s) sticky %s (%s) fixed %s | take all %s takeable %s sticky %s fixed %s" % (a["rows"]["all"][A], a["rows"]["takeable"][A], a["rows"]["takeable"]["true behaviour given"],
              a["rows"]["sticky"][A], a["rows"]["sticky"]["true behaviour given"], a["rows"]["fixed"][A], a["take"]["all"][A], a["take"]["takeable"][A], a["take"]["sticky"][A], a["take"]["fixed"][A]))
