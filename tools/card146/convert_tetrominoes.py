"""Tetrominoes TFRecords -> the .npz files AKOrN's code reads, without TensorFlow.

Follows akorn/data/convert_tfrecord_to_np.py: crop 35x35 to 32x32, label = index of the
mask that covers the pixel (0 is the background), first 64 records 'eval', next 320 'test',
next 10,000 'val', next 60,000 'train'.
usage: python tools/card146/convert_tetrominoes.py data/tetrominoes
"""
import gzip, struct, sys, os
import numpy as np


def varint(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1
        v |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return v, i


def fields(b):
    """Yield (field number, payload) for the length-delimited fields of a message."""
    i = 0
    while i < len(b):
        tag, i = varint(b, i)
        assert tag & 7 == 2, tag
        n, i = varint(b, i)
        yield tag >> 3, b[i:i + n]
        i += n


def one_byte_list(b):
    """A BytesList whose values are single bytes: (0x0A, 0x01, value) repeated."""
    a = np.frombuffer(b, np.uint8).reshape(-1, 3)
    assert (a[:, 0] == 0x0A).all() and (a[:, 1] == 1).all()
    return a[:, 2]


def parse(rec):
    out = {}
    (_, feats), = fields(rec)
    for _, entry in fields(feats):
        kv = dict(fields(entry))
        key = bytes(kv[1]).decode()
        if key in ("image", "mask"):
            (kind, payload), = fields(kv[2])
            assert kind == 1
            out[key] = one_byte_list(payload)
    return out


def main(root):
    sizes = [("eval", 64), ("test", 320), ("val", 10000), ("train", 60000)]
    total = sum(n for _, n in sizes)
    images = np.zeros((total, 3, 32, 32), np.float32)
    labels = np.zeros((total, 32, 32), np.uint8)
    with gzip.open(os.path.join(root, "tetrominoes_train.tfrecords"), "rb") as f:
        for k in range(total):
            head = f.read(12)
            n, = struct.unpack("<Q", head[:8])
            rec = memoryview(f.read(n)); f.read(4)
            d = parse(rec)
            img = d["image"].reshape(35, 35, 3)[1:33, 1:33]
            mask = d["mask"].reshape(4, 35, 35)
            lab = np.zeros((35, 35), np.uint8)
            for idx in range(4):
                lab[mask[idx] == 255] = idx
            images[k] = img.transpose(2, 0, 1) / 255.0
            labels[k] = lab[1:33, 1:33]
            if (k + 1) % 10000 == 0:
                print(k + 1, flush=True)
    s = 0
    for name, n in sizes:
        np.savez_compressed(os.path.join(root, f"tetrominoes_{name}.npz"), images=images[s:s + n], labels=labels[s:s + n])
        s += n
    print("objects per image (test):", np.bincount([len(np.unique(l)) - 1 for l in labels[64:384]]))


if __name__ == "__main__":
    main(sys.argv[1])
