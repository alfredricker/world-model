"""The least-cost one-to-one assignment (the Hungarian method with potentials, O(n^2 m)), for rectangular costs.
Returns (rows, cols) as scipy.optimize.linear_sum_assignment does; scipy is not a dependency of this project."""
import numpy as np


def linear_sum_assignment(cost):
    C = np.asarray(cost, np.float64)
    if C.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    flip = C.shape[0] > C.shape[1]
    if flip:
        C = C.T
    n, m = C.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, np.int64)                      # p[j]: row assigned to column j (1-based; 0 = none)
    way = np.zeros(m + 1, np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = C[i0 - 1] - u[i0] - v[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows, cols = np.array(rows, np.int64), np.array(cols, np.int64)
    if flip:
        rows, cols = cols, rows
    o = np.argsort(rows)
    return rows[o], cols[o]
