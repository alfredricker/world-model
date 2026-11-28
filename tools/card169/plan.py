"""Card 169: Zero-heuristic event-level BFS, spoiler guard, and P14 surprise recovery over frozen SceneWNet
in live OGBench visual-scene-v0 (256-px frames + proprioception).

Modes:
  --gate 1                                 run feasibility gate on self-made problems ONLY -> runs/169/gate.json
  --practice_all 1                         run 150 self-made practice problems (50 dev + 50 mix + 50 put) -> runs/169/own.json
  --task <1..5> --n 50 --seed <s0>         live benchmark run with frozen SceneWNet & frozen own.json -> runs/169/plan_<tag>.json
  --summarize 1                            compile 3-seed x 5-task results + ablations -> experiments/169-.../results.json
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import ogbench  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from encode import load_encoder  # noqa: E402
from train import features  # noqa: E402
import importlib.util  # noqa: E402

# Card 168's model.py is loaded by path: earlier cards put their own `model` module first on sys.path.
_spec = importlib.util.spec_from_file_location("card168_model", Path(__file__).resolve().parents[1] / "card168/model.py")
M168 = importlib.util.module_from_spec(_spec)
sys.modules["card168_model"] = M168
_spec.loader.exec_module(M168)

ROOT = Path(__file__).resolve().parents[2]
DATA = Path.home() / ".ogbench/data"
OUT = ROOT / "runs/169"
CEN = np.array([0.425, 0.0, 0.0], dtype=np.float32)
R, LIMIT, HOLD = 256, 750, 3
EP, PRE, POST, W, CLEAR = 1001, 10, 10, 10, 2
BETA, TOP = 1.0, 10

arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


class Done(Exception):
    pass


def touches_from_contact(c):
    """Exact contact-bracketed touch segmentation (tools/card120/touch.py: W=10, CLEAR=2)."""
    m = c > 0.05
    p = np.r_[False, m, False]
    d = np.diff(p.astype(np.int8))
    a, b = np.where(d == 1)[0], np.where(d == -1)[0]
    st, en = [], []
    for x, y in zip(a, b):
        if st and x - en[-1] < 3 and x // EP == st[-1] // EP:
            en[-1] = y
        else:
            st.append(x)
            en.append(y)
    st, en = np.array(st), np.array(en) - 1
    ok = (st // EP == en // EP) & (st % EP >= W + CLEAR - 1) & (en % EP <= EP - 1 - W - CLEAR + 1)
    return st[ok], en[ok]


class TouchMemory:
    """Self-contained proprioceptive trajectory store from scene-play-v0 m400 (episodes 400-499)."""
    def __init__(self, first=400):
        z = np.load(DATA / "scene-play-v0.npz")
        sl = slice(first * EP, (first + 100) * EP)
        self.obs = z["observations"][sl]
        self.act = z["actions"][sl]
        self.st, self.en = touches_from_contact(self.obs[:, 18])
        self.inside = (self.st % EP >= PRE + 1) & (self.en % EP <= EP - POST - 2)

        same = np.arange(len(self.obs) - 1) % EP != EP - 1
        d = (self.obs[1:, 12:15] - self.obs[:-1, 12:15])[same]
        self.gain = np.linalg.lstsq(self.act[:-1, :3][same], d, rcond=None)[0]
        yaw = np.arctan2(self.obs[:, 16], self.obs[:, 15])
        dy = np.angle(np.exp(1j * (yaw[1:] - yaw[:-1])))[same]
        self.yaw_gain = float(np.linalg.lstsq(self.act[:-1, 3:4][same], dy, rcond=None)[0][0])
        dg = (self.obs[1:, 17] - self.obs[:-1, 17])[same]
        self.grip_gain = float(np.linalg.lstsq(self.act[:-1, 4:5][same], dg, rcond=None)[0][0])
        self.safe_z = float(np.quantile(self.obs[:, 14], 0.9))

    def pose(self, t):
        o = self.obs[t]
        return np.r_[o[12:15], np.arctan2(o[16], o[15]), o[17]]


def action_to(mem, cur, tgt, move=(1, 1, 1)):
    a = np.zeros(5, np.float32)
    want = (tgt[:3] - cur[:3]) * np.array(move)
    a[:3] = np.linalg.solve(mem.gain.T, want)
    a[3] = np.angle(np.exp(1j * (tgt[3] - cur[3]))) / mem.yaw_gain
    a[4] = (tgt[4] - cur[4]) / mem.grip_gain
    return np.clip(a, -1.0, 1.0)


class Runner:
    def __init__(self, env):
        self.env, self.u = env, env.unwrapped

    def reset(self, **options):
        self.frame, info = self.env.reset(options=options)
        self.goal, self.steps, self.success = info["goal"], 0, False
        return info

    def pose(self):
        o = self.u.compute_ob_info()
        return np.r_[
            (o["proprio/effector_pos"] - CEN) * 10.0,
            o["proprio/effector_yaw"][0],
            o["proprio/gripper_opening"][0] * 3.0,
        ]

    def step(self, a):
        self.frame, _, _, _, info = self.env.step(a)
        self.steps += 1
        if info["success"]:
            self.success = True
            raise Done
        if self.steps >= LIMIT:
            raise Done

    def goto(self, mem, goal, move=(1, 1, 1), tol=0.05, most=60):
        for _ in range(most):
            cur = self.pose()
            if np.abs((goal[:3] - cur[:3]) * np.array(move)).max() < tol:
                break
            g = goal.copy()
            if move == (0, 0, 1):
                g[3:] = cur[3:]
            self.step(action_to(mem, cur, g, move))

    def lift(self, mem):
        cur = self.pose()
        self.goto(mem, np.r_[cur[:2], mem.safe_z, cur[3:]], (0, 0, 1))

    def rest(self, mem, pose):
        self.lift(mem)
        up = pose.copy()
        up[2] = max(mem.safe_z, pose[2])
        self.goto(mem, up)
        self.goto(mem, pose)
        for _ in range(HOLD):
            self.step(np.zeros(5, np.float32))

    def truth(self):
        c = self.u._data.joint("object_joint_0").qpos[:3].copy()
        dr = float(self.u._data.joint("drawer_slide").qpos[0])
        wi = float(self.u._data.joint("window_slide").qpos[0])
        lk = [int(x) for x in self.u._cur_button_states]
        in_dr = bool((abs(c[0] - 0.33) < 0.12) and (c[2] > 0.04) and (c[1] < -0.10))
        where = ("hidden" if dr > -0.05 else "open") if in_dr else "table"
        return dict(locks=lk, drawer=round(dr, 4), window=round(wi, 4), cube=[round(float(x), 4) for x in c], where=where)


def follow(run, mem, st, en, off0, off1):
    def tgt(t):
        p = mem.pose(t).copy()
        w = np.clip((t - st) / max(en - st, 1), 0.0, 1.0)
        p[:3] += (1.0 - w) * off0 + w * off1
        return p

    first = tgt(st - PRE)
    up = first.copy()
    up[2] = mem.safe_z
    run.lift(mem)
    run.goto(mem, up)
    run.goto(mem, first)
    for t in range(st - PRE + 1, en + POST + 1):
        run.step(action_to(mem, run.pose(), tgt(t)))


class SceneBFSAgent:
    """System 2 Event-Level BFS + Working-Memory Belief + Spoiler Guard + P14 Surprise over frozen SceneWNet."""
    def __init__(self, rng, ckpt_path=None, own_path=None, practice=False, truth_oracle=False):
        self.rng = rng
        self.practice = practice
        self.truth_oracle = truth_oracle
        self.dev = "cuda"

        self.mem = TouchMemory(first=400)
        self.enc = load_encoder(self.dev)
        pca = torch.load(ROOT / "runs/132/pca.pt", map_location=self.dev, weights_only=False)
        self.pca_mu = pca["mu"].to(self.dev).float()
        self.pca_sd = pca["sd"].to(self.dev).float()
        self.pca_V = pca["V"].to(self.dev).float()
        self.pca_scale = pca["scale"].to(self.dev).float()

        ck = torch.load(ckpt_path or (ROOT / "runs/168/scene_wnet_0.pt"), map_location=self.dev, weights_only=False)
        self.meta = ck["meta"]
        self.var_map = ck["var_map"]  # used ONLY if self.truth_oracle is True (gate upper bound)
        self.net = M168.SceneWNet(n_slots=ck["n_slots"], mov_si=ck["mov_si"]).to(self.dev)
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()

        self.n_slots = ck["n_slots"]
        self.mov_si = ck["mov_si"]
        self.slots = self.meta["slots"]
        self.kinds = self.meta["slot_kind"]
        self.cuts = {int(k): float(v) for k, v in self.meta["slot_cuts"].items()}
        self.z_cut = float(self.meta["z_cut"])

        # Encode m400 touch stream strictly with SceneWNet (ZERO truth fields read!)
        Z_m = np.load(ROOT / "runs/132/z_m400.npy").astype(np.float32)
        ev_m = np.load(ROOT / "runs/132/ev_m400.npz")
        tc_m = ev_m["touch"].astype(np.float32)
        ep_m = ev_m["episode"].astype(np.int64)
        assert len(self.mem.st) == len(Z_m), (len(self.mem.st), len(Z_m))
        assert np.allclose(self.mem.obs[self.mem.st + 2, 12:15], tc_m[:, :3], atol=1e-4)

        self.H_m, self.Val_m, self.V_m, self.Aim_m = M168.encode_all(self.net, Z_m)
        self.u_act_m = M168.make_u_act(tc_m, self.meta)

        pl_m = np.linalg.norm(tc_m[:, None, :3] - self.meta["cen"][None], axis=-1).argmin(1)
        grp_m = self.meta["group"][pl_m]
        same_ep = ep_m[:-1] == ep_m[1:]
        ins = self.mem.inside

        # Rest pose: farthest pre-touch pose from all active slot contact centers (og3/og4 rule)
        centres = np.stack([
            self.mem.obs[self.mem.st[grp_m == u] + 2, 12:15].mean(0) for u in self.slots
        ])
        cand_r = np.where(ins)[0]
        P_r = self.mem.obs[self.mem.st[cand_r] - 2, 12:15]
        best_r = cand_r[int(np.argmax(np.linalg.norm(P_r[:, None] - centres[None], axis=-1).min(1)))]
        self.rest = self.mem.pose(self.mem.st[best_r] - 2)
        self.rest[4] = float(np.quantile(self.mem.obs[:, 17], 0.9))

        # Build skill library from SceneWNet encodings on m400 (zero truth)
        self.lock_sis = [k for k, kd in enumerate(self.kinds) if kd == "lock"]
        self.slide_sis = [k for k, kd in enumerate(self.kinds) if kd == "slide"]

        # 1. Lock presses & manifold prototypes
        self.press_cands = {}
        self.press_u = {}
        self.lock_proto = {}
        for c_k in self.lock_sis:
            self.press_cands[c_k] = {}
            self.press_u[c_k] = {}
            self.lock_proto[c_k] = {}
            for b in (0, 1):
                m_b = (self.Val_m[:, c_k] > 0.0) if b == 1 else (self.Val_m[:, c_k] <= 0.0)
                proto = self.H_m[m_b, c_k].mean(0)
                self.lock_proto[c_k][b] = (proto / (np.linalg.norm(proto) + 1e-8)).astype(np.float32)

                flips = (
                    ins[:-1] & same_ep & (grp_m[:-1] == self.slots[c_k])
                    & (((self.Val_m[:-1, c_k] > 0.0).astype(int)) == b)
                    & (((self.Val_m[1:, c_k] > 0.0).astype(int)) == (1 - b))
                )
                for s_k in self.slide_sis:
                    flips &= np.abs(self.Val_m[1:, s_k] - self.Val_m[:-1, s_k]) <= self.cuts[s_k]
                idx_f = np.where(flips)[0]
                self.press_cands[c_k][b] = idx_f
                self.press_u[c_k][b] = np.median(self.u_act_m[idx_f], axis=0).astype(np.float32)

        # 2. Slide pulls (two directions per slide slot) & manifold prototypes
        self.slide_ext = {}
        self.slide_cands = {}
        self.slide_u = {}
        any_lock_flip = np.zeros(len(tc_m) - 1, bool)
        for c_k in self.lock_sis:
            any_lock_flip |= same_ep & ((self.Val_m[1:, c_k] > 0.0) != (self.Val_m[:-1, c_k] > 0.0))

        for s_k in self.slide_sis:
            ext_lo, ext_hi = [float(x) for x in np.quantile(self.Val_m[:, s_k], [0.03, 0.97])]
            mid = 0.5 * (ext_lo + ext_hi)
            self.slide_ext[s_k] = dict(lo=ext_lo, hi=ext_hi, mid=mid, span=max(ext_hi - ext_lo, 1e-3))
            base_s = ins[:-1] & same_ep & (grp_m[:-1] == self.slots[s_k]) & (~any_lock_flip)
            dv = self.Val_m[1:, s_k] - self.Val_m[:-1, s_k]
            idx_lo = np.where(base_s & (dv < -1.2 * self.cuts[s_k]))[0]
            idx_hi = np.where(base_s & (dv > 1.2 * self.cuts[s_k]))[0]
            self.slide_cands[s_k] = dict(lo=idx_lo, hi=idx_hi)

            full_lo = idx_lo[(self.Val_m[idx_lo, s_k] > mid) & (self.Val_m[idx_lo + 1, s_k] < mid)]
            full_hi = idx_hi[(self.Val_m[idx_hi, s_k] < mid) & (self.Val_m[idx_hi + 1, s_k] > mid)]
            self.slide_u[s_k] = dict(
                lo=np.median(self.u_act_m[full_lo if len(full_lo) else idx_lo], axis=0).astype(np.float32),
                hi=np.median(self.u_act_m[full_hi if len(full_hi) else idx_hi], axis=0).astype(np.float32),
            )

        # 3. Movable moves: followed table moves + Card 143 un-followed / interior release-endpoint moves
        mov_u = self.slots[self.mov_si]
        disp_m = np.linalg.norm(tc_m[:, 3:6] - tc_m[:, :3], axis=1)
        own_mov = np.where(grp_m == mov_u)[0]
        m_idx, m_A, m_B, m_put = [], [], [], []
        for j_pos, a_i in enumerate(own_mov):
            if not ins[a_i] or disp_m[a_i] <= 0.25:
                continue
            A_pt = self.mem.obs[self.mem.st[a_i] + 2, 12:15]
            rel_pt = self.mem.obs[self.mem.en[a_i], 12:15]
            # A release above z_cut counts as a put-in only if SceneWNet then reads the movable slot at interior
            # height (26 of 55 high releases in m400 left the cube on the table, which emptied put_req).
            read_in = bool(
                a_i + 1 < len(tc_m) and ep_m[a_i + 1] == ep_m[a_i]
                and self.V_m[a_i + 1] >= 0.5 and self.Aim_m[a_i + 1, 2] > self.z_cut
            )
            if rel_pt[2] > self.z_cut and read_in:
                B_pt, is_p = rel_pt, True
            elif j_pos + 1 < len(own_mov) and ep_m[own_mov[j_pos + 1]] == ep_m[a_i]:
                b_i = own_mov[j_pos + 1]
                # Use next contact point unless a slide moved in between
                inter_slide = any(grp_m[k_i] in [self.slots[s] for s in self.slide_sis] for k_i in range(a_i + 1, b_i))
                B_pt, is_p = (rel_pt if inter_slide else self.mem.obs[self.mem.st[b_i] + 2, 12:15]), False
            else:
                B_pt, is_p = rel_pt, False
            if np.linalg.norm(B_pt - A_pt) * 0.1 > self.cuts[self.mov_si]:
                m_idx.append(a_i); m_A.append(A_pt); m_B.append(B_pt); m_put.append(is_p)

        self.mov_idx = np.array(m_idx, dtype=np.int64)
        self.mov_A = np.stack(m_A).astype(np.float32)
        self.mov_B = np.stack(m_B).astype(np.float32)
        self.mov_is_put = np.array(m_put, dtype=bool)
        self.med_open0 = float(np.median(tc_m[self.mov_idx, 6]))

        tab_mask = ~self.mov_is_put
        self.tab_idx = self.mov_idx[tab_mask]
        self.tab_A = self.mov_A[tab_mask]
        self.tab_B = self.mov_B[tab_mask]
        self.table_lo = self.tab_A.min(0)[:2] * 0.1 + CEN[:2] + 0.02
        self.table_hi = self.tab_A.max(0)[:2] * 0.1 + CEN[:2] - 0.02

        self.put_idx = self.mov_idx[self.mov_is_put]
        self.put_A = self.mov_A[self.mov_is_put]
        self.put_B = self.mov_B[self.mov_is_put]
        self.B_cont_med = np.median(self.put_B, axis=0).astype(np.float32)

        # Empirical start-state applicability for interior placements (GOAL P5 / Konidaris 2018; zero object names!)
        self.put_req = {}
        for s_k in self.slide_sis:
            q02, q98 = [float(x) for x in np.quantile(self.Val_m[self.put_idx, s_k], [0.02, 0.98])]
            if (q98 - q02) < 0.55 * self.slide_ext[s_k]["span"]:
                v_med = float(np.median(self.Val_m[self.put_idx, s_k]))
                self.put_req[s_k] = (q02 - self.cuts[s_k], q98 + self.cuts[s_k], v_med)

        self.own_file = Path(own_path) if own_path else (OUT / "own.json")
        self.own = {int(k): v for k, v in json.load(open(self.own_file)).items()} if self.own_file.exists() else {}
        self.reset_episode()

    def reset_episode(self):
        self._ep_tag = None
        self.failed = set()
        self.pending = None
        self.bel_in_cont = False
        self.bel_last_aim = None
        self.lock_override = {}          # c_k -> forced 0/1 after P14 surprise
        self.slide_fail_streak = {}      # s_k -> consecutive non-moving pulls
        self.refuted_ops = set()         # P14 episode refutations

    def snap_slide_h(self, s_k, v_target):
        idx16 = np.argsort(np.abs(self.Val_m[:, s_k] - v_target))[:16]
        h = self.H_m[idx16, s_k].mean(0)
        return (h / (np.linalg.norm(h) + 1e-8)).astype(np.float32)

    @torch.no_grad()
    def encode_frame(self, frame, truth=None):
        rgb = torch.from_numpy(np.ascontiguousarray(frame))[None].to(self.dev)
        f = features(self.enc, rgb)                                              # [1, 1024, 768]
        z = (((f - self.pca_mu) / self.pca_sd) @ self.pca_V) / self.pca_scale    # [1, 1024, 64]
        eo = self.net.encode(z)
        h = eo["h"][0].cpu().numpy()                                             # [S, D]
        val = eo["val"][0].cpu().numpy()                                         # [S]
        vis_logit = float(eo["vis_logit"][0].item())
        p_vis = float(1.0 / (1.0 + np.exp(-vis_logit)))
        aim = eo["aim"][0].cpu().numpy()                                         # [3]

        if self.truth_oracle and truth is not None:
            # Declared Gate Upper Bound ONLY: map simulator truth into SceneWNet's slot space
            dr_si, wi_si = self.var_map["drawer"], self.var_map["window"]
            l0_si, l1_si = self.var_map["lock0"], self.var_map["lock1"]
            s_op, s_cl = float(self.meta["s_open"]), float(self.meta["s_closed"])
            val[dr_si] = s_cl + (float(-truth["drawer"]) / 0.16) * (s_op - s_cl)
            h[dr_si] = self.snap_slide_h(dr_si, val[dr_si])

            # Orient window extreme using correlation on m400's two extremes:
            # open window (0.2) vs closed (0.0) mapped via SceneWNet's encoded frame sign on current vs prototype
            w_lo, w_hi = self.slide_ext[wi_si]["lo"], self.slide_ext[wi_si]["hi"]
            # Notice on gate/init frames, SceneWNet window over10 is 97.4% accurate, so align direction once:
            if not hasattr(self, "_wi_sign"):
                ev_tmp = np.load(ROOT / "runs/132/ev_m400.npz")
                self._wi_open_v = float(np.median(self.Val_m[ev_tmp["window"] > 0.15, wi_si]))
                self._wi_shut_v = float(np.median(self.Val_m[ev_tmp["window"] < 0.02, wi_si]))
            val[wi_si] = self._wi_shut_v + (float(truth["window"]) / 0.20) * (self._wi_open_v - self._wi_shut_v)
            h[wi_si] = self.snap_slide_h(wi_si, val[wi_si])

            for lk_idx, l_si in [(0, l0_si), (1, l1_si)]:
                b = int(truth["locks"][lk_idx])
                h[l_si] = self.lock_proto[l_si][b]
                val[l_si] = 2.5 if b == 1 else -2.5

            c_xyz = np.array(truth["cube"], dtype=np.float32)
            aim = (c_xyz - CEN) * 10.0
            if truth["where"] == "hidden":
                p_vis, vis_logit = 0.02, -4.0
            else:
                p_vis, vis_logit = 0.98, 4.0

        return dict(h=h, val=val, vis_logit=vis_logit, p_vis=p_vis, aim=aim)

    def own_prob(self, i, key):
        tot = [v for v in self.own.values() if v[2] == key]
        p0 = (sum(v[0] for v in tot) + 0.5) / (sum(v[1] for v in tot) + 1.0)
        s, n, _ = self.own.get(i, (0, 0, key))
        return (s + BETA * p0) / (n + BETA)

    def pick_touch(self, cands, key):
        left = [int(i) for i in cands if int(i) not in self.failed] or [int(i) for i in cands]
        return max(left, key=lambda i: (self.own_prob(i, key), -left.index(i)))

    def is_cont_closed(self, val):
        for s_k, (lo_ok, hi_ok, v_open) in self.put_req.items():
            if lo_ok <= val[s_k] <= hi_ok:
                return False
        return True

    def goal_met(self, st, goal):
        for c_k in self.lock_sis:
            if (st["val"][c_k] > 0.0) != (goal["val"][c_k] > 0.0):
                return False
        for s_k in self.slide_sis:
            if abs(st["val"][s_k] - goal["val"][s_k]) > self.cuts[s_k]:
                return False
        if goal["p_vis"] < 0.5:
            if not (st["p_vis"] < 0.5 and st["in_cont"]):
                return False
        else:
            if st["p_vis"] < 0.5:
                return False
            if np.linalg.norm(st["aim"] - goal["aim"]) * 0.1 > self.cuts[self.mov_si]:
                return False
        return True

    def state_key(self, st, goal):
        lk = tuple(int(st["val"][c] > 0.0) for c in self.lock_sis)
        sl = tuple(int(round(st["val"][s] / max(self.cuts[s], 0.15))) for s in self.slide_sis)
        if st["p_vis"] < 0.5 and st["in_cont"]:
            mv = "hidden_in_cont"
        elif st["in_cont"]:
            mv = "open_in_cont"
        elif goal["p_vis"] >= 0.5 and np.linalg.norm(st["aim"] - goal["aim"]) * 0.1 <= self.cuts[self.mov_si]:
            mv = "at_goal"
        else:
            mv = "at_start"
        return (lk, sl, mv)

    def candidate_ops(self, st, goal, nocond=False, relax=False):
        """Enumerate candidate macro-actions from imagined state `st`, applying empirical applicability & spoiler guard."""
        ops = []
        need_put_chain = (goal["p_vis"] < 0.5) and not (st["p_vis"] < 0.5 and st["in_cont"])

        # 1. Movable-slot actions
        if goal["p_vis"] >= 0.5:
            if st["p_vis"] >= 0.5 and np.linalg.norm(st["aim"] - goal["aim"]) * 0.1 > self.cuts[self.mov_si]:
                ok_app = True
                if not nocond and goal["aim"][2] > self.z_cut:
                    ok_app = all(lo <= st["val"][s_k] <= hi for s_k, (lo, hi, _) in self.put_req.items())
                if ok_app:
                    u = np.r_[st["aim"], self.med_open0, goal["aim"], 1.0].astype(np.float32)
                    ops.append(dict(kind="move_goal", slot=self.mov_si, tgt=goal["aim"].copy(), u_act=u))
        else:
            # Goal wants target absent / put away in container
            if st["p_vis"] >= 0.5 and not st["in_cont"]:
                ok_app = True if nocond else all(lo <= st["val"][s_k] <= hi for s_k, (lo, hi, _) in self.put_req.items())
                ctx_sig = tuple(int(round(st["val"][s_k] / 0.25)) for s_k in self.put_req)
                if ok_app and ("put_in", ctx_sig) not in self.refuted_ops:
                    u = np.r_[st["aim"], self.med_open0, self.B_cont_med, 1.0].astype(np.float32)
                    ops.append(dict(kind="put_in", slot=self.mov_si, tgt=self.B_cont_med.copy(), u_act=u, ctx=ctx_sig))

        # 2. Slide-slot actions (with Spoiler Guard: never close a container while `put_in` still needs it open!)
        for s_k in self.slide_sis:
            targets = []
            # If container needs to open first for put_in (or to reveal hidden cube when goal_vis >= 0.5):
            if s_k in self.put_req:
                lo_ok, hi_ok, v_open = self.put_req[s_k]
                need_open_for_put = (goal["p_vis"] < 0.5) and (not st["in_cont"]) and not (lo_ok <= st["val"][s_k] <= hi_ok)
                need_open_for_reveal = (goal["p_vis"] >= 0.5) and (st["p_vis"] < 0.5) and not (lo_ok <= st["val"][s_k] <= hi_ok)
                if not nocond and (need_open_for_put or need_open_for_reveal):
                    targets.append(v_open)

            # Pull toward goal value unless Spoiler Guard forbids closing container before target is inside!
            if abs(st["val"][s_k] - goal["val"][s_k]) > self.cuts[s_k]:
                spoil_put = (
                    (not nocond) and (s_k in self.put_req) and (goal["p_vis"] < 0.5)
                    and (not st["in_cont"])
                    and not (self.put_req[s_k][0] <= goal["val"][s_k] <= self.put_req[s_k][1])
                )
                if not spoil_put:
                    targets.append(float(goal["val"][s_k]))

            seen_dirs = set()
            for tv in targets:
                if abs(tv - st["val"][s_k]) <= self.cuts[s_k]:
                    continue
                d_tag = "hi" if tv > st["val"][s_k] else "lo"
                if d_tag in seen_dirs:
                    continue
                # P14: a pull that failed twice under these lock readings is not retried until a lock changes
                if ("slide", s_k, d_tag, tuple(int(st["val"][c] > 0.0) for c in self.lock_sis)) in self.refuted_ops:
                    continue
                seen_dirs.add(d_tag)
                ops.append(dict(kind="slide", slot=s_k, dir=d_tag, tgt=float(tv), u_act=self.slide_u[s_k][d_tag]))

        # 3. Lock-slot presses (with Spoiler Guard: never disable a lock while its slide still needs to move!)
        if not nocond:
            for c_k in self.lock_sis:
                b_now = int(st["val"][c_k] > 0.0)
                # Check if disabling (1 -> 0) would spoil a pending slide move:
                if b_now == 1 and (goal["val"][c_k] <= 0.0):
                    spoils_slide = False
                    for s_k, cause_c in self.meta["cause_of"].items():
                        if int(cause_c) == c_k:
                            if abs(st["val"][s_k] - goal["val"][s_k]) > self.cuts[s_k]:
                                spoils_slide = True
                            if (s_k in self.put_req) and need_put_chain:
                                spoils_slide = True
                    if spoils_slide and not relax:
                        continue
                ops.append(dict(kind="press", slot=c_k, b_from=b_now, tgt=1 - b_now, u_act=self.press_u[c_k][b_now]))
        else:
            # In --nocond 1 (greedy depth-1 ablation), only press a lock if the lock itself differs from goal
            for c_k in self.lock_sis:
                b_now = int(st["val"][c_k] > 0.0)
                b_goal = int(goal["val"][c_k] > 0.0)
                if b_now != b_goal:
                    ops.append(dict(kind="press", slot=c_k, b_from=b_now, tgt=b_goal, u_act=self.press_u[c_k][b_now]))

        return ops

    @torch.no_grad()
    def bfs_plan(self, root_st, goal_st, max_depth=8, max_exp=250, nocond=False, relax=False):
        """Batched Event-Level BFS in SceneWNet's shared latent space (zero pixel rollouts)."""
        t0 = time.perf_counter()
        if self.goal_met(root_st, goal_st):
            return None, 0, round((time.perf_counter() - t0) * 1000.0, 3)

        if nocond:
            ops = self.candidate_ops(root_st, goal_st, nocond=True)
            ms = round((time.perf_counter() - t0) * 1000.0, 3)
            if not ops:
                return None, 1, ms
            # Greedy 1-step without condition/consequence check: pick a differing variable directly (slides/moves before lock restore)
            non_press = [o for o in ops if o["kind"] != "press"]
            chosen = non_press[int(self.rng.integers(len(non_press)))] if non_press else ops[0]
            return [chosen], 1, ms

        visited = {self.state_key(root_st, goal_st)}
        frontier = [(root_st, [])]
        expansions = 0
        best_partial = None
        best_partial_score = self._progress_score(root_st, goal_st)

        for depth in range(max_depth):
            if not frontier or expansions >= max_exp:
                break
            batch_nodes = []
            batch_ops = []
            batch_chains = []
            for st_node, chain in frontier:
                if expansions >= max_exp:
                    break
                expansions += 1
                for op in self.candidate_ops(st_node, goal_st, nocond=False, relax=relax):
                    batch_nodes.append(st_node)
                    batch_ops.append(op)
                    batch_chains.append(chain + [op])

            if not batch_ops:
                break

            ht = torch.from_numpy(np.stack([n["h"] for n in batch_nodes])).to(self.dev)
            vt = torch.tensor([n["vis_logit"] for n in batch_nodes], dtype=torch.float32, device=self.dev)
            at = torch.from_numpy(np.stack([n["aim"] for n in batch_nodes])).to(self.dev)
            ut = torch.from_numpy(np.stack([o["u_act"] for o in batch_ops])).to(self.dev)

            tr = self.net.forward_transition(ht, vt, at, ut)
            p_chg = tr["p_chg"].cpu().numpy()                                    # [B, S]
            vis_next_l = tr["vis_next_logit"].cpu().numpy()                      # [B]
            aim_next = tr["aim_next"].cpu().numpy()                              # [B, 3]

            next_frontier = []
            for b_i, (st_node, op, new_chain) in enumerate(zip(batch_nodes, batch_ops, batch_chains)):
                if p_chg[b_i, op["slot"]] < 0.5:
                    continue  # Blocked precondition predicted by SceneWNet!

                nh = st_node["h"].copy()
                nval = st_node["val"].copy()
                nvis_l = float(st_node["vis_logit"])
                np_vis = float(st_node["p_vis"])
                naim = st_node["aim"].copy()
                nin_cont = bool(st_node["in_cont"])

                if op["kind"] == "press":
                    c_k = op["slot"]
                    nb = int(op["tgt"])
                    nh[c_k] = self.lock_proto[c_k][nb]
                    nval[c_k] = 2.5 if nb == 1 else -2.5
                elif op["kind"] == "slide":
                    s_k = op["slot"]
                    tv = float(op["tgt"])
                    nh[s_k] = self.snap_slide_h(s_k, tv)
                    nval[s_k] = tv
                    # Cross-variable occlusion / reveal predicted by SceneWNet!
                    if nin_cont:
                        nvis_l = float(vis_next_l[b_i])
                        np_vis = float(1.0 / (1.0 + np.exp(-nvis_l)))
                elif op["kind"] == "move_goal":
                    naim = op["tgt"].copy()
                    nin_cont = bool(naim[2] > self.z_cut)
                    nvis_l, np_vis = 4.0, 0.98
                elif op["kind"] == "put_in":
                    naim = aim_next[b_i].copy()
                    if naim[2] <= self.z_cut:
                        naim = self.B_cont_med.copy()
                    nin_cont = True
                    nvis_l = float(max(vis_next_l[b_i], 1.0))  # visible in open container until container closes
                    np_vis = float(1.0 / (1.0 + np.exp(-nvis_l)))

                nxt_st = dict(h=nh, val=nval, vis_logit=nvis_l, p_vis=np_vis, aim=naim, in_cont=nin_cont)
                if self.goal_met(nxt_st, goal_st):
                    ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    return new_chain, expansions, ms

                s_key = self.state_key(nxt_st, goal_st)
                if s_key not in visited:
                    visited.add(s_key)
                    next_frontier.append((nxt_st, new_chain))
                    sc = self._progress_score(nxt_st, goal_st)
                    if sc > best_partial_score:
                        best_partial_score = sc
                        best_partial = new_chain

            frontier = next_frontier

        ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return best_partial, expansions, ms

    def _progress_score(self, st, goal):
        sc = 0.0
        if goal["p_vis"] < 0.5:
            if st["p_vis"] < 0.5 and st["in_cont"]:
                sc += 3.0
            elif st["in_cont"]:
                sc += 1.5
        else:
            if st["p_vis"] >= 0.5 and np.linalg.norm(st["aim"] - goal["aim"]) * 0.1 <= self.cuts[self.mov_si]:
                sc += 2.0
        for s_k in self.slide_sis:
            if abs(st["val"][s_k] - goal["val"][s_k]) <= self.cuts[s_k]:
                sc += 1.0
        for c_k in self.lock_sis:
            if (st["val"][c_k] > 0.0) == (goal["val"][c_k] > 0.0):
                sc += 0.4
        return sc

    def instantiate_skill(self, op, now_st):
        """Select concrete remembered trajectory i + offsets (off0, off1) via own-tries rule."""
        zero = np.zeros(3, dtype=np.float32)
        if op["kind"] == "press":
            c_k, b_from = op["slot"], op["b_from"]
            key = f"press_{c_k}"
            cands = self.rng.permutation(self.press_cands[c_k][b_from])
            i = self.pick_touch(cands, key)
            return i, zero, zero, key

        if op["kind"] == "slide":
            s_k, d_tag, tv = op["slot"], op["dir"], op["tgt"]
            key = f"slide_{s_k}_{d_tag}"
            pool = self.slide_cands[s_k][d_tag]
            cost = np.abs(self.Val_m[pool, s_k] - now_st["val"][s_k]) + np.abs(self.Val_m[pool + 1, s_k] - tv)
            top_cands = pool[np.argsort(cost)[:TOP]]
            i = self.pick_touch(top_cands, key)
            return i, zero, zero, key

        if op["kind"] == "move_goal":
            key = f"move_{self.mov_si}"
            an, ag = now_st["aim"], op["tgt"]
            use_put = bool(ag[2] > self.z_cut)
            idx_p = self.put_idx if use_put else self.tab_idx
            A_p = self.put_A if use_put else self.tab_A
            B_p = self.put_B if use_put else self.tab_B
            order = np.argsort(np.linalg.norm(A_p - an, axis=1) + np.linalg.norm(B_p - ag, axis=1))[:TOP]
            at = {int(idx_p[j]): int(j) for j in order}
            i = self.pick_touch(list(at.keys()), key)
            j = at[i]
            return i, (an - A_p[j]).astype(np.float32), (ag - B_p[j]).astype(np.float32), key

        if op["kind"] == "put_in":
            key = "put_in"
            an = now_st["aim"]
            cost = np.linalg.norm(self.put_A - an, axis=1) + 0.6 * np.linalg.norm(self.put_B - self.B_cont_med, axis=1)
            order = np.argsort(cost)[:TOP]
            at = {int(self.put_idx[j]): int(j) for j in order}
            i = self.pick_touch(list(at.keys()), key)
            j = at[i]
            off0 = (an - self.put_A[j]).astype(np.float32)
            off1 = np.clip(self.B_cont_med - self.put_B[j], -0.25, 0.25).astype(np.float32)
            return i, off0, off1, key

        raise ValueError(op)

    def decide(self, now_raw, goal_raw, nocond=False):
        """Update working-memory belief & P14 surprise from last touch, then run Event-Level BFS."""
        now = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in now_raw.items()}
        goal = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in goal_raw.items()}

        # Initialize episode working-memory belief on first step
        if self._ep_tag is None:
            self._ep_tag = True
            self.bel_in_cont = bool(now["p_vis"] < 0.5) or bool(now["aim"][2] > self.z_cut)
            if now["p_vis"] >= 0.5:
                self.bel_last_aim = now["aim"].copy()
        else:
            if now["p_vis"] >= 0.5:
                self.bel_in_cont = bool(now["aim"][2] > self.z_cut)
                self.bel_last_aim = now["aim"].copy()
            elif self.bel_last_aim is not None and not self.bel_in_cont:
                # Target became transiently un-read on table without entering container -> keep last table aim
                now["aim"] = self.bel_last_aim.copy()

        # Apply any active P14 lock-state overrides before evaluating outcome & planning
        raw_lock_b = {c_k: int(now["val"][c_k] > 0.0) for c_k in self.lock_sis}

        # Evaluate pending touch outcome -> update own-tries + P14 surprise recovery!
        last_ok = None
        p14_fired = None
        if self.pending is not None:
            p_op, p_i, p_key, prev_st = self.pending
            kind = p_op["kind"]
            if kind == "press":
                c_k = p_op["slot"]
                # Clear override on c_k since we just physically pressed c_k!
                self.lock_override.pop(c_k, None)
                last_ok = (raw_lock_b[c_k] != p_op["b_from"]) or (p_op["c_was_overridden"])
            elif kind == "slide":
                s_k, tv = p_op["slot"], p_op["tgt"]
                reached = abs(now["val"][s_k] - tv) <= self.cuts[s_k]
                moved = abs(now["val"][s_k] - prev_st["val"][s_k]) > self.cuts[s_k]
                last_ok = bool(reached)
                if moved:
                    self.slide_fail_streak[s_k] = 0
                    c_cause = self.meta["cause_of"].get(s_k)
                    if c_cause is not None:
                        self.lock_override.pop(int(c_cause), None)
                else:
                    self.slide_fail_streak[s_k] = self.slide_fail_streak.get(s_k, 0) + 1
                    # P14 surprise: if slide did not budge despite SceneWNet reading lock enabled, refute lock reading!
                    c_cause = self.meta["cause_of"].get(s_k)
                    if not nocond and c_cause is not None and self.slide_fail_streak[s_k] >= 2:
                        c_k = int(c_cause)
                        self.lock_override[c_k] = 0
                        self.slide_fail_streak[s_k] = 0
                        ctx = tuple(
                            int(self.lock_override.get(c, raw_lock_b[c])) for c in self.lock_sis
                        )
                        self.refuted_ops.add(("slide", s_k, p_op["dir"], ctx))
                        p14_fired = f"slide_{s_k}_blocked->override_lock_{c_k}=0"
            elif kind == "move_goal":
                last_ok = bool(now["p_vis"] >= 0.5 and np.linalg.norm(now["aim"] - p_op["tgt"]) * 0.1 <= self.cuts[self.mov_si])
            elif kind == "put_in":
                # Succeeded if target reached interior height z > z_cut OR disappeared into container back shadow!
                entered = bool((now["p_vis"] >= 0.5 and now["aim"][2] > self.z_cut) or (now["p_vis"] < 0.5))
                last_ok = entered
                if entered:
                    self.bel_in_cont = True
                elif not nocond:
                    # P14 surprise: put_in left target on table -> check if container opening was insufficient
                    # Refute only a put-in chosen with its container condition unmet; with the condition met the
                    # miss is a failed touch, which own-tries (self.failed) already excludes.
                    for s_k, (lo_ok, hi_ok, v_open) in self.put_req.items():
                        if not (lo_ok <= prev_st["val"][s_k] <= hi_ok):
                            self.refuted_ops.add(("put_in", p_op.get("ctx")))
                            p14_fired = f"put_in_rim->reopen_slide_{s_k}"

            if not last_ok:
                self.failed.add(int(p_i))
            if self.practice:
                s_cnt, n_cnt, _ = self.own.get(int(p_i), (0, 0, p_key))
                self.own[int(p_i)] = [int(s_cnt + int(bool(last_ok))), int(n_cnt + 1), p_key]
                OUT.mkdir(parents=True, exist_ok=True)
                json.dump(self.own, open(self.own_file, "w"))
            self.pending = None

        # Apply active P14 lock overrides to `now` state for BFS
        for c_k, forced_b in self.lock_override.items():
            now["h"][c_k] = self.lock_proto[c_k][forced_b]
            now["val"][c_k] = 2.5 if forced_b == 1 else -2.5

        now["in_cont"] = bool(self.bel_in_cont)
        goal["in_cont"] = bool(goal["p_vis"] < 0.5) or bool(goal["aim"][2] > self.z_cut)

        chain, expansions, bfs_ms = self.bfs_plan(now, goal, max_depth=8, max_exp=250, nocond=nocond)
        if not chain and not nocond and not self.goal_met(now, goal):
            chain, exp2, ms2 = self.bfs_plan(now, goal, max_depth=8, max_exp=250, relax=True)
            expansions, bfs_ms = expansions + exp2, round(bfs_ms + ms2, 3)
        log = dict(
            p_vis_now=round(now["p_vis"], 3),
            p_vis_goal=round(goal["p_vis"], 3),
            bel_in_cont=self.bel_in_cont,
            val_now=[round(float(x), 2) for x in now["val"]],
            val_goal=[round(float(x), 2) for x in goal["val"]],
            last_worked=last_ok,
            p14=p14_fired,
            expansions=expansions,
            bfs_ms=bfs_ms,
            plan_len=len(chain) if chain else 0,
            plan=[o["kind"] + f"({o['slot']})" for o in chain] if chain else [],
        )
        if not chain:
            return None, log

        op0 = chain[0]
        op0["c_was_overridden"] = bool(op0["kind"] == "press" and op0["slot"] in self.lock_override)
        i, off0, off1, key = self.instantiate_skill(op0, now)
        self.pending = (op0, i, key, now)
        return (op0["kind"], op0["slot"], i, off0, off1), log


# --- Self-made problem generators (NEVER read benchmark tasks 1..5!) ---

def make_dev_problem(u, rng):
    while True:
        t = copy.deepcopy(u.task_infos[0])
        for side in ("init", "goal"):
            t[side]["button_states"] = rng.integers(0, 2, 2)
            t[side]["drawer_pos"] = float(rng.choice([0.0, -0.16]))
            t[side]["window_pos"] = float(rng.choice([0.0, 0.2]))
        t["goal"]["block_xyzs"] = t["init"]["block_xyzs"].copy()
        i, g = t["init"], t["goal"]
        if (i["button_states"] != g["button_states"]).any() or i["drawer_pos"] != g["drawer_pos"] or i["window_pos"] != g["window_pos"]:
            return t


def make_mix_problem(u, rng, lo, hi):
    t = make_dev_problem(u, rng)
    while True:
        c, g = rng.uniform(lo, hi), rng.uniform(lo, hi)
        if np.linalg.norm(c - g) > 0.08:
            break
    t["init"]["block_xyzs"] = np.array([[*c, 0.02]])
    t["goal"]["block_xyzs"] = np.array([[*g, 0.02]])
    return t


def make_locked_put_problem(u, rng, lo, hi, hidden_spots):
    """Self-made locked put-away problem: drawer starts locked & closed (0.0) with cube at random on table;
    goal has drawer closed (0.0) with cube hidden inside (drawn from m400 hidden spots), random lock/window states."""
    t = copy.deepcopy(u.task_infos[0])
    w_init = float(rng.choice([0.0, 0.2]))
    w_goal = w_init if rng.random() < 0.7 else float(rng.choice([0.0, 0.2]))
    b1_init = int(rng.integers(0, 2))
    b0_goal = int(rng.integers(0, 2))
    b1_goal = b1_init if w_goal == w_init else int(rng.integers(0, 2))

    t["init"]["button_states"] = np.array([0, b1_init])
    t["init"]["drawer_pos"] = 0.0
    t["init"]["window_pos"] = w_init
    lo_clear = np.maximum(lo, np.array([-1.0, -0.05]))
    t["init"]["block_xyzs"] = np.array([[*rng.uniform(lo_clear, hi), 0.02]])

    t["goal"]["button_states"] = np.array([b0_goal, b1_goal])
    t["goal"]["drawer_pos"] = 0.0
    t["goal"]["window_pos"] = w_goal
    t["goal"]["block_xyzs"] = hidden_spots[int(rng.integers(len(hidden_spots)))][None].astype(np.float64)
    return t


def load_hidden_spots_for_problem_gen():
    """Used ONLY by make_locked_put_problem to place a valid physical goal cube inside the closed drawer."""
    ev = np.load(ROOT / "runs/132/ev_m400.npz")
    c = ev["cube"]
    in_shut = (np.abs(c[:, 0] - 0.33) < 0.06) & (c[:, 2] > 0.04) & (c[:, 1] < -0.28) & (ev["drawer"] > -0.02)
    return c[in_shut]


def run_episodes(ag, env, mode, n, seed, nocond=False, max_touches=18):
    rng = np.random.default_rng(seed)
    ag.rng = rng
    env.reset(seed=seed)
    run = Runner(env)
    hidden_spots = load_hidden_spots_for_problem_gen() if mode == "put" else None
    cube_fn = lambda: run.u._data.joint("object_joint_0").qpos[:3].copy()

    eps = []
    all_bfs_ms = []
    all_expansions = []
    p14_count = 0

    for e in range(n):
        ag.reset_episode()
        if isinstance(mode, int):
            run.reset(task_id=mode)
        elif mode == "dev":
            run.reset(task_info=make_dev_problem(run.u, rng))
        elif mode == "mix":
            run.reset(task_info=make_mix_problem(run.u, rng, ag.table_lo, ag.table_hi))
        elif mode == "put":
            run.reset(task_info=make_locked_put_problem(run.u, rng, ag.table_lo, ag.table_hi, hidden_spots))
        else:
            raise ValueError(mode)

        t_info = run.u.cur_task_info
        g_true_cube = np.array(t_info["goal"]["block_xyzs"][0])
        g_in_dr = bool((abs(g_true_cube[0] - 0.33) < 0.12) and (g_true_cube[2] > 0.04) and (g_true_cube[1] < -0.10))
        g_truth = dict(
            locks=[int(x) for x in t_info["goal"]["button_states"]],
            drawer=float(t_info["goal"]["drawer_pos"]),
            window=float(t_info["goal"]["window_pos"]),
            cube=[float(x) for x in g_true_cube],
            where=("hidden" if float(t_info["goal"]["drawer_pos"]) > -0.05 else "open") if g_in_dr else "table",
        )
        goal_st = ag.encode_frame(run.goal, truth=g_truth)
        ep_log = dict(start=run.truth(), goal=g_truth, touches=[])

        try:
            while len(ep_log["touches"]) < max_touches:
                run.rest(ag.mem, ag.rest)
                cur_tr = run.truth()
                now_st = ag.encode_frame(run.frame, truth=cur_tr)
                act, dlog = ag.decide(now_st, goal_st, nocond=nocond)
                all_bfs_ms.append(dlog["bfs_ms"])
                all_expansions.append(dlog["expansions"])
                if dlog["p14"]:
                    p14_count += 1
                ep_log["touches"].append(dict(
                    at=run.steps,
                    act=[act[0], int(act[1]), int(act[2])] if act else None,
                    truth=cur_tr,
                    cube_cm=round(float(np.linalg.norm(cube_fn() - g_true_cube) * 100.0), 1),
                    **dlog,
                ))
                if act is None:
                    break
                follow(run, ag.mem, ag.mem.st[act[2]], ag.mem.en[act[2]], act[3], act[4])
        except Done:
            pass

        end_tr = run.truth()
        ep_log.update(
            success=bool(run.success),
            steps=int(run.steps),
            end=end_tr,
            cube_end_cm=round(float(np.linalg.norm(cube_fn() - g_true_cube) * 100.0), 1),
            cube_in_closed_drawer=bool(end_tr["where"] == "hidden"),
            gave_up=bool(ep_log["touches"] and ep_log["touches"][-1]["act"] is None),
        )
        eps.append(ep_log)
        acts_short = [a["act"][0] + str(a["act"][1]) if a["act"] else "stop" for a in ep_log["touches"]]
        print(f"[{mode} s={seed} e={e:02d}] {'WIN' if run.success else 'FAIL'} {run.steps:3d}st {acts_short} cube={ep_log['cube_end_cm']}cm end={end_tr['where']}/{end_tr['drawer']}/{end_tr['window']}/{end_tr['locks']}", flush=True)

    succ = round(100.0 * float(np.mean([x["success"] for x in eps])), 2)
    win_steps = [x["steps"] for x in eps if x["success"]]
    return dict(
        mode=str(mode),
        seed=seed,
        episodes=n,
        nocond=nocond,
        success_pct=succ,
        cube_hidden_pct=round(100.0 * float(np.mean([x["cube_in_closed_drawer"] for x in eps])), 2),
        cube_within_4cm_pct=round(100.0 * float(np.mean([x["cube_end_cm"] <= 4.0 for x in eps])), 2),
        mean_steps_when_success=round(float(np.mean(win_steps)), 1) if win_steps else None,
        mean_touches=round(float(np.mean([len(x["touches"]) for x in eps])), 2),
        mean_bfs_ms=round(float(np.mean(all_bfs_ms)), 3) if all_bfs_ms else 0.0,
        p95_bfs_ms=round(float(np.quantile(all_bfs_ms, 0.95)), 3) if all_bfs_ms else 0.0,
        mean_expansions=round(float(np.mean(all_expansions)), 2) if all_expansions else 0.0,
        p14_recoveries=int(p14_count),
        log=eps,
    )


def run_gate():
    OUT.mkdir(parents=True, exist_ok=True)
    env = gym.make("visual-scene-v0", terminate_at_goal=False, width=R, height=R)

    # 1. Upper bound: SceneBFS with simulator truth on 30 locked put-away problems
    ag_oracle = SceneBFSAgent(np.random.default_rng(101), own_path=OUT / "nonexistent.json", practice=False, truth_oracle=True)
    ub = run_episodes(ag_oracle, env, "put", 30, seed=101, nocond=False)

    # 2. Trivial baseline: --nocond 1 on the exact same 30 locked put-away problems
    ag_base = SceneBFSAgent(np.random.default_rng(101), own_path=OUT / "nonexistent.json", practice=False, truth_oracle=False)
    tb = run_episodes(ag_base, env, "put", 30, seed=101, nocond=True)

    # 3. Gate bar 1: --mix 30 (random locks + slides + table cube)
    mx = run_episodes(ag_base, env, "mix", 30, seed=102, nocond=False)

    # 4. Gate bar 2: 30 locked put-away problems (visual SceneWNet, zero truth)
    lp = run_episodes(ag_base, env, "put", 30, seed=101, nocond=False)

    gate_summary = {
        "upper_bound_truth_locked_put30": {k: v for k, v in ub.items() if k != "log"},
        "trivial_baseline_nocond_locked_put30": {k: v for k, v in tb.items() if k != "log"},
        "mix30": {k: v for k, v in mx.items() if k != "log"},
        "locked_put30": {k: v for k, v in lp.items() if k != "log"},
    }
    json.dump(gate_summary, open(OUT / "gate.json", "w"), indent=2)
    print("\n=== CARD 169 GATE SUMMARY ===")
    print(json.dumps(gate_summary, indent=2), flush=True)
    os._exit(0)


def run_practice_all():
    """Run 150 self-made practice problems (50 dev + 50 mix + 50 locked put-away, matching Card 130's 150-ep budget)."""
    OUT.mkdir(parents=True, exist_ok=True)
    own_f = OUT / "own.json"
    if own_f.exists():
        own_f.unlink()
    env = gym.make("visual-scene-v0", terminate_at_goal=False, width=R, height=R)
    ag = SceneBFSAgent(np.random.default_rng(31), own_path=own_f, practice=True, truth_oracle=False)
    r_dev = run_episodes(ag, env, "dev", 50, seed=31, nocond=False)
    r_mix = run_episodes(ag, env, "mix", 50, seed=32, nocond=False)
    r_put = run_episodes(ag, env, "put", 50, seed=33, nocond=False)
    prac_log = {
        "dev50": {k: v for k, v in r_dev.items() if k != "log"},
        "mix50": {k: v for k, v in r_mix.items() if k != "log"},
        "put50": {k: v for k, v in r_put.items() if k != "log"},
        "distinct_touches_tried": len(ag.own),
        "total_tries": int(sum(v[1] for v in ag.own.values())),
    }
    json.dump(prac_log, open(OUT / "practice_summary.json", "w"), indent=2)
    print("\n=== PRACTICE SUMMARY ===")
    print(json.dumps(prac_log, indent=2), flush=True)
    os._exit(0)


def summarize_benchmark():
    seeds = [0, 1, 2]
    tasks = [1, 2, 3, 4, 5]
    per_task = {}
    bfs_ms_all = []
    exp_all = []
    for t in tasks:
        vals, steps_l, ms_l, ex_l, p14_l = [], [], [], [], []
        for s in seeds:
            d = json.load(open(OUT / f"plan_task{t}_s{s}.json"))
            vals.append(d["success_pct"])
            if d["mean_steps_when_success"] is not None:
                steps_l.append(d["mean_steps_when_success"])
            ms_l.append(d["mean_bfs_ms"])
            ex_l.append(d["mean_expansions"])
            p14_l.append(d["p14_recoveries"])
        bfs_ms_all.extend(ms_l)
        exp_all.extend(ex_l)
        per_task[f"task{t}"] = dict(
            mean_pct=round(float(np.mean(vals)), 2),
            std_pct=round(float(np.std(vals)), 2),
            per_seed_pct=vals,
            seed0_pct=vals[0],
            mean_steps_when_success=round(float(np.mean(steps_l)), 1) if steps_l else None,
            mean_bfs_ms=round(float(np.mean(ms_l)), 3),
            mean_expansions=round(float(np.mean(ex_l)), 2),
            total_p14_recoveries=int(sum(p14_l)),
        )

    five_mean_per_seed = [round(float(np.mean([per_task[f"task{t}"]["per_seed_pct"][si] for t in tasks])), 2) for si in range(len(seeds))]
    t13_mean = round(float(np.mean([per_task[f"task{t}"]["mean_pct"] for t in (1, 2, 3)])), 2)
    five_mean = round(float(np.mean(five_mean_per_seed)), 2)

    nc2 = json.load(open(OUT / "plan_task2_nocond_s0.json"))
    nc4 = json.load(open(OUT / "plan_task4_nocond_s0.json"))

    res = dict(
        card="169",
        seeds=seeds,
        episodes_per_task_per_seed=50,
        total_benchmark_episodes=750,
        per_task=per_task,
        tasks_1_to_3_mean_pct=t13_mean,
        five_task_mean_pct=five_mean,
        five_task_mean_per_seed_pct=five_mean_per_seed,
        seed0_five_task_mean_pct=five_mean_per_seed[0],
        overall_mean_bfs_ms=round(float(np.mean(bfs_ms_all)), 3),
        overall_mean_expansions=round(float(np.mean(exp_all)), 2),
        nocond_ablation_seed0=dict(
            task2_pct=nc2["success_pct"],
            task4_pct=nc4["success_pct"],
        ),
        comparisons=dict(
            og4_seed0=dict(tasks=[100.0, 86.0, 82.0, 0.0, 0.0], tasks_1_to_3_mean=89.33, five_task_mean=53.6),
            gas_published=dict(five_task_mean=54.4),
            hiql_published=dict(tasks=[80.0, 81.0, 61.0, 20.0, 3.0], five_task_mean=49.0),
            gcivl_published=dict(tasks=[84.0, 24.0, 16.0, 0.0, 0.0], five_task_mean=24.8),
        ),
    )
    out_p = ROOT / "experiments/169-zero-heuristic-event-bfs-on-ogbench/results.json"
    json.dump(res, open(out_p, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


def main():
    if int(arg("--gate", 0)):
        run_gate()
    if int(arg("--practice_all", 0)):
        run_practice_all()
    if int(arg("--summarize", 0)):
        summarize_benchmark()
        return

    OUT.mkdir(parents=True, exist_ok=True)
    task = int(arg("--task", 0))
    dev = int(arg("--dev", 0))
    mix = int(arg("--mix", 0))
    put = int(arg("--put", 0))
    n = dev or mix or put or int(arg("--n", 50))
    seed = int(arg("--seed", 0))
    nocond = bool(int(arg("--nocond", 0)))
    practice = bool(int(arg("--practice", 0)))
    ckpt = arg("--ckpt", str(ROOT / "runs/168/scene_wnet_0.pt"))

    mode = task if task else ("dev" if dev else ("mix" if mix else "put"))
    default_tag = f"task{task}{'_nocond' if nocond else ''}_s{seed}" if task else f"{mode}{n}{'_nocond' if nocond else ''}_s{seed}"
    tag = arg("--tag", default_tag)

    env = gym.make("visual-scene-v0", terminate_at_goal=False, width=R, height=R)
    ag = SceneBFSAgent(np.random.default_rng(seed), ckpt_path=ckpt, own_path=OUT / "own.json", practice=practice)
    res = run_episodes(ag, env, mode, n, seed=seed, nocond=nocond)
    res["tag"] = tag
    json.dump(res, open(OUT / f"plan_{tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
