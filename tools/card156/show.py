"""compact view of runs/156 files: show.py <files...>"""
import sys, json
A = "appearance and own transitions"
for f in sys.argv[1:]:
    for r in json.load(open(f)):
        print("%s world %d by %s, %ds: moved %d of %d movable objects; %d patterns (%d still); objects with transitions %s; classes %d %s purity %s" % (r["arm"], r["world"], r["by"], r["seconds"], r["objects_moved_in_training"],
              r["movable_objects_in_training"], r["patterns"], r["patterns_of_still_groups"], r["objects_by_true_behaviour"], r["classes"], r["class_sizes"], r["class_purity"]))
        for part in ("new_objects", "known_objects"):
            a = r[part]
            print("   %-13s found movable %s (takeable %s sticky %s fixed %s) ground labels %s | take all %s (takeable %s sticky %s fixed %s) | effects all %s (bound %s)" % (part, a["found"]["movable"]["found"], a["found"]["takeable"]["found"],
                  a["found"]["sticky"]["found"], a["found"]["fixed"]["found"], a["labels_on_the_ground_per_scene"], a["take"]["all"][A], a["take"]["takeable"][A], a["take"]["sticky"][A], a["take"]["fixed"][A], a["rows"]["all"][A], a["rows"]["all"]["true behaviour given"]))
