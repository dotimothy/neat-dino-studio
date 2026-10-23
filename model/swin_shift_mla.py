"""Plain <-> shifted Swin window layouts with Slice + Concat only (no reshape), for the MLA.

Tokens are in window layout (1, C, nW, 144): windows column-major (index b * nh + a),
tokens row-major in the 12x12 window. A shifted window (a', b') of the map rolled by -6
is made of four 6x6 quadrants of plain windows:
  top-left  <- (a',   b'  ) bottom-right     top-right    <- (a',   b'+1) bottom-left
  bottom-left <- (a'+1, b') top-right        bottom-right <- (a'+1, b'+1) top-left
(indices mod nh / nw). The reverse move is the mirror image.
"""
from __future__ import annotations

import torch

WS, H = 12, 6


def _quad(x, qi, qj):
    """Quadrant (qi, qj in {0, 1}) of every window: (1, C, nW, 36), row-major."""
    rows = [x[:, :, :, (qi * H + i) * WS + qj * H: (qi * H + i) * WS + qj * H + H] for i in range(H)]
    return torch.cat(rows, 3)


def _assemble(tl, tr, bl, br):
    """Four (1, C, nW, 36) quadrants -> (1, C, nW, 144) windows."""
    rows = []
    for i in range(H):
        rows += [tl[:, :, :, i * H:(i + 1) * H], tr[:, :, :, i * H:(i + 1) * H]]
    for i in range(H):
        rows += [bl[:, :, :, i * H:(i + 1) * H], br[:, :, :, i * H:(i + 1) * H]]
    return torch.cat(rows, 3)


def _shift_windows(q, nh, nw, da, db):
    """Window (a, b) <- window (a + da, b + db) (mod), windows indexed b * nh + a."""
    if db:
        s = (db % nw) * nh                         # whole columns: a roll of the window axis
        q = torch.cat([q[:, :, s:], q[:, :, :s]], 2)
    if da:
        cols = []
        for b in range(nw):                       # inside each column, roll by da
            c = q[:, :, b * nh:(b + 1) * nh]
            s = da % nh
            cols.append(torch.cat([c[:, :, s:], c[:, :, :s]], 2) if s else c)
        q = torch.cat(cols, 2)
    return q


def plain_to_shifted(x, nh, nw):
    br, bl, tr, tl = _quad(x, 1, 1), _quad(x, 1, 0), _quad(x, 0, 1), _quad(x, 0, 0)
    return _assemble(br, _shift_windows(bl, nh, nw, 0, 1), _shift_windows(tr, nh, nw, 1, 0),
                     _shift_windows(tl, nh, nw, 1, 1))


def shifted_to_plain(y, nh, nw):
    tl, tr, bl, br = _quad(y, 0, 0), _quad(y, 0, 1), _quad(y, 1, 0), _quad(y, 1, 1)
    return _assemble(_shift_windows(br, nh, nw, -1, -1), _shift_windows(bl, nh, nw, -1, 0),
                     _shift_windows(tr, nh, nw, 0, -1), tl)
