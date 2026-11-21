"""compact view of runs/154 files: show.py <files...>"""
import sys, json
for f in sys.argv[1:]:
    for r in json.load(open(f)):
        m = r["memory"]; print("world %d by %s, %d episodes, %ds: %d patterns; %d pool objects moved (%s%% with a good pattern), %d never moved" % (r["world"], r["by"], r["train_episodes"], r["seconds"], m["patterns"], m["pool_objects_that_moved"], m["pool_moved_with_a_good_pattern"], m["pool_objects_never_moved"]))
        for part in ("first_sight", "covered"):
            for arm, v in r[part].items():
                if "all" not in v: print("   %-11s %-34s n %d found %s" % (part, arm, v["n"], v["found"])); continue
                print("   %-11s %-34s" % (part, arm), " | ".join("%s: n %d found %s itself %s" % (k.split(" hidden")[0], v[k]["n"], v[k]["found"], v[k]["itself"]) for k in v if isinstance(v[k], dict) and "itself" in v[k] and (part == "covered" or k == "all")),
                      "| not known: wrongly recognised %s, found %s | ground labels %s" % (v["not known"]["wrongly_recognised"], v["not known"]["found"], v["labels_on_the_ground_per_scene"]))
        for arm, v in r["moving"].items(): print("   moving      %-34s" % arm, v)
