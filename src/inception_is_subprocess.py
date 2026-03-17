from __future__ import annotations

import sys
import threading
import struct
import pickle
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


_WORKER_SCRIPT = r"""\
import sys, struct, pickle, os
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision.models import inception_v3, Inception_V3_Weights
except Exception as exc:
    sys.stdout.buffer.write(("ERROR " + str(exc) + "\n").encode())
    sys.stdout.buffer.flush()
    raise

try:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False, aux_logits=False)
    model.eval()
    sys.stdout.buffer.write(b"READY\n")
except Exception as exc:
    sys.stdout.buffer.write(("ERROR " + str(exc) + "\n").encode())
sys.stdout.buffer.flush()

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

while True:
    hdr = _read_exact(8)
    if hdr is None:
        break
    (n,) = struct.unpack("<Q", hdr)
    payload = _read_exact(n)
    if payload is None:
        break
    imgs = pickle.loads(payload)
    # imgs: uint8 NHWC (B,H,W,3) or float32 [0,1]
    try:
        arr = np.asarray(imgs)
        if arr.dtype == np.uint8:
            x = torch.from_numpy(arr).float() / 255.0
        else:
            x = torch.from_numpy(arr).float()
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2)
        # Resize to 299x299
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        logits_np = logits.cpu().numpy().astype(np.float32)
        probs_np = probs.cpu().numpy().astype(np.float32)
        out = ("ok", logits_np, probs_np)
    except Exception as exc:
        out = ("error", str(exc))
    out_bytes = pickle.dumps(out, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(struct.pack("<Q", len(out_bytes)) + out_bytes)
    sys.stdout.buffer.flush()
"""


@dataclass
class InceptionISResult:
    logits: "object"  # np.ndarray float32 [B,1000]
    probs: "object"   # np.ndarray float32 [B,1000]


class InceptionISSubprocess:
    """Long-running torchvision Inception-v3 worker (no torch import in parent).

    Protocol:
      - Worker prints 'READY\\n' or 'ERROR ...\\n' to stdout at startup.
      - Request: 8-byte little-endian length + pickle(images)
      - Reply:   8-byte length + pickle(('ok', logits, probs) | ('error', msg))
    """

    def __init__(self):
        import subprocess

        self._stderr_tail = deque(maxlen=50)
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        ready = self._proc.stdout.readline().decode(errors="replace").strip()
        if ready != "READY":
            tail = "\n".join(self._stderr_tail)
            raise RuntimeError(
                "Inception IS worker failed to start.\n"
                f"stdout: {ready!r}\n"
                f"stderr_tail:\n{tail}"
            )

    def _drain_stderr(self):
        try:
            for line in iter(self._proc.stderr.readline, b""):
                self._stderr_tail.append(line.decode(errors="replace").rstrip())
        except Exception:
            return

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("Inception IS worker died unexpectedly (stdout closed).")
            buf += chunk
        return buf

    def infer(self, images) -> InceptionISResult:
        if self._proc.poll() is not None:
            tail = "\n".join(self._stderr_tail)
            raise RuntimeError(f"Inception IS worker is not running.\nstderr_tail:\n{tail}")
        data = pickle.dumps(images, protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(struct.pack("<Q", len(data)) + data)
        self._proc.stdin.flush()
        hdr = self._read_exact(8)
        (n,) = struct.unpack("<Q", hdr)
        payload = self._read_exact(n)
        out = pickle.loads(payload)
        if not out or out[0] != "ok":
            msg = out[1] if out and out[0] == "error" else "unknown error"
            tail = "\n".join(self._stderr_tail)
            raise RuntimeError(f"Inception IS worker error: {msg}\nstderr_tail:\n{tail}")
        _, logits, probs = out
        return InceptionISResult(logits=logits, probs=probs)

    def shutdown(self):
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

