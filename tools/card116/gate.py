"""Card 116, gate steps 2-3: the ceiling of the 14 x 14 patch grid, and the k-means baseline.

- Ceiling: each patch takes its true majority label (from the 64 x 64 truth), upsampled as slot masks are.
- k-means: 8 clusters on raw DINOv2 patch vectors (200,000 patches sampled from training frames), each cluster a
  segment.

  bin/prun python tools/card116/gate.py [--grid 32] → runs/116/gate.json (gate32.json at 32 x 32, card 116.1)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).parent))
from metrics import onehot, patch_of_pixels, scores, upsample  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/116'


def main():
    G = int(sys.argv[sys.argv.index('--grid') + 1]) if '--grid' in sys.argv else 14
    t = np.load(OUT / 'truth.npz')
    frames, labels = t['frames'], t['labels']
    pid = patch_of_pixels(G).reshape(-1)
    maj = np.zeros((len(labels), G * G), np.int64)
    for i, lab in enumerate(labels.reshape(len(labels), -1)):
        counts = np.zeros((G * G, 7), np.int64)
        np.add.at(counts, (pid, lab), 1)
        maj[i] = counts.argmax(1)
    res = {'ceiling': scores(upsample(onehot(maj, 7)), labels)}
    print('ceiling', res['ceiling'], flush=True)
    ftr = np.load(ROOT / 'runs/114/feat_train.npy' if G == 14 else OUT / 'feat448_train.npy', mmap_mode='r')
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(len(ftr), 2000, replace=False))
    X = np.asarray(ftr[rows]).reshape(-1, 384).astype(np.float32)
    X = X[rng.choice(len(X), 200000, replace=False)]
    km = MiniBatchKMeans(8, random_state=0, batch_size=4096, n_init=3).fit(X)
    if G == 14:
        fva = np.load(ROOT / 'runs/114/feat_val.npy', mmap_mode='r')
        rows_v = frames
    else:
        fva = np.load(OUT / 'feat448_val.npy', mmap_mode='r')
        rows_v = np.searchsorted(np.load(OUT / 'frames448_val.npy'), frames)
    ids = km.predict(np.asarray(fva[rows_v]).reshape(-1, 384).astype(np.float32)).reshape(len(frames), G * G)
    res['kmeans'] = scores(upsample(onehot(ids, 8)), labels)
    print('kmeans', res['kmeans'])
    json.dump(res, open(OUT / ('gate.json' if G == 14 else f'gate{G}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
