"""Card 058: trace the familiar-world episodes that fail with walking that prices clearing the way.

Runs card 053's planner check through card 058's runner, and for every failed layout prints the last steps: the
action chosen and the head of its chain of conditions (evaluator names).

  bin/prun python tools/card058/trace.py runs/054/b_m0.5_399.pt --worlds key --layouts 30
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shorter as SH                                   # noqa: E402

MV, CM = SH.MV, SH.CM
S7 = CM.S7
VP = MV.VP


def install_trace():
    choose0 = S7.Plan047.choose

    def choose(self, st):
        res = choose0(self, st)
        log = self.__dict__.setdefault("log", [])
        log.append((int(st[1]), None if res is None else int(res.action),
                    None if res is None else [str(c)[:48] for c in res.trace[:5]]))
        return res

    S7.Plan047.choose = choose
    job0 = VP._act_job

    def act_job(job):
        n0 = len(MV.PLANS)
        out = job0(job)
        for (i, rec), pl in zip(enumerate(out), MV.PLANS[n0:]):
            if not rec["done"]:
                print("FAILED layout", job[0][i], "steps", rec["steps"], "random", rec["random"], flush=True)
                for row in getattr(pl, "log", [])[-12:]:
                    print("   ", row, flush=True)
        return out

    VP._act_job = act_job


def main():
    args = sys.argv[1:]
    get = lambda k, d: next((args[i + 1] for i in range(len(args) - 1) if args[i] == k), d)
    w2_install = SH.walk2.install

    def install_all():
        w2_install()
        SH.install()

    SH.walk2.install = install_all
    cr_main = SH.MV.CR.main

    def main_traced():
        install_trace()
        cr_main()

    SH.MV.CR.main = main_traced
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "card053"))
    import planner_check as PC                         # noqa: E402
    sys.argv = ["planner_check.py", args[0], "--gates", "0", "--codes", "noise", "@@", "--dev", "--arm", "A",
                "--seeds", "399-399", "--layouts", get("--layouts", "30"), "--worlds", get("--worlds", "key"),
                "--out", "runs/058/trace.json"]
    PC.main()


if __name__ == "__main__":
    main()
