"""print runs of card 153 compactly: show.py runs/153/<file>.json ..."""
import sys, json
for f in sys.argv[1:]:
    for d in json.load(open(f)):
        if "new_objects" in d:
            a = d["new_objects"]["rows"]["all"]; t = d["new_objects"]["take"]["all"]; k = d["known_objects"]; x = k["rows"].get("exceptions", {}); xt = k["take"].get("exceptions", {})
            print("%-28s world %2d %-6s n %3d | kinds %d %s purity %5.1f | looks %s | effects %5.1f bound %5.1f one %5.1f | take %5.1f (%5.1f, %5.1f) | exceptions eff %s vs looks %s, take %s vs %s | %ds" % (
                f.split("/")[-1][:-5], d["world"], d["by"], d["train_episodes"], d["kinds"], d["kind_sizes"], d["kind_purity"], d["looks_share"], a["looks only"], a["true behaviour given"], a["one kind for all"],
                t["looks only"], t["true behaviour given"], t["one kind for all"], x.get("looks and own records"), x.get("looks only"), xt.get("looks and own records"), xt.get("looks only"), d["seconds"]))
        else:
            a = d["new_words"]; g = lambda k: a[k]["nats"]; e = d["known_exceptions"]
            print("%-28s world %3d %-6s n %4d | words %d (%d met <= twice) | kinds %d %s purity %5.1f | looks %s | nats: module %.4f reader-of-forms %.4f true class %.4f one %.4f no-kinds %.4f | gap closed %5.1f (%5.1f) | exceptions %.4f vs form only %.4f, bound %.4f | %ds" % (
                f.split("/")[-1][:-5], d["world"], d["by"], d["train_sentences"], d["words_met"], d["words_met_at_most_twice"], d["kinds"], d["kind_sizes"], d["kind_purity"], d["looks_share"],
                g("the module: kind from the form"), g("the true rule, with the share of known words that break it"), g("true class given"), g("one kind for all"), g("no kinds, form coupled to the context"),
                d["gap_closed"], d["gap_closed_to_true_class_given"], e["form and the word's own records"]["nats"], e["form only"]["nats"], e["true class given"]["nats"], d["seconds"]))
