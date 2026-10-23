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

WS, H = 12, 6            # Swin-B (grounding-dino-base); Swin-T uses ws 7, shift 3


def _region(x, r0, r1, c0, c1, ws):
    """Rows r0..r1-1, columns c0..c1-1 of every window: (1, C, nW, (r1-r0)*(c1-c0)), row-major."""
    return torch.cat([x[:, :, :, r * ws + c0: r * ws + c1] for r in range(r0, r1)], 3)


def _assemble2(tl, tr, bl, br, h1, w1, h2, w2):
    """Four blocks (top h1 rows: tl is w1 wide, tr w2; bottom h2 rows: bl, br) -> windows."""
    rows = []
    for i in range(h1):
        rows += [tl[:, :, :, i * w1:(i + 1) * w1], tr[:, :, :, i * w2:(i + 1) * w2]]
    for i in range(h2):
        rows += [bl[:, :, :, i * w1:(i + 1) * w1], br[:, :, :, i * w2:(i + 1) * w2]]
    return torch.cat(rows, 3)


def _quad(x, qi, qj):
    """Quadrant (qi, qj in {0, 1}) of every 12x12 window: (1, C, nW, 36), row-major."""
    return _region(x, qi * H, qi * H + H, qj * H, qj * H + H, WS)


def _assemble(tl, tr, bl, br):
    """Four (1, C, nW, 36) quadrants -> (1, C, nW, 144) windows."""
    return _assemble2(tl, tr, bl, br, H, H, H, H)


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


def plain_to_shifted(x, nh, nw, ws=WS, shift=H):
    """Rolled by -shift, a shifted window (a, b) takes the last ws-shift rows/columns of plain
    window (a, b) and the first shift rows/columns of its neighbours (a+1, b), (a, b+1), (a+1, b+1)."""
    k = ws - shift
    tl = _region(x, shift, ws, shift, ws, ws)                                    # k x k
    tr = _shift_windows(_region(x, shift, ws, 0, shift, ws), nh, nw, 0, 1)       # k x shift
    bl = _shift_windows(_region(x, 0, shift, shift, ws, ws), nh, nw, 1, 0)       # shift x k
    br = _shift_windows(_region(x, 0, shift, 0, shift, ws), nh, nw, 1, 1)        # shift x shift
    return _assemble2(tl, tr, bl, br, k, k, shift, shift)


def shifted_to_plain(y, nh, nw, ws=WS, shift=H):
    """The reverse: a plain window's first shift rows/columns come from the shifted windows
    before it, the rest from shifted window (a, b)."""
    k = ws - shift
    tl = _shift_windows(_region(y, k, ws, k, ws, ws), nh, nw, -1, -1)            # shift x shift
    tr = _shift_windows(_region(y, k, ws, 0, k, ws), nh, nw, -1, 0)              # shift x k
    bl = _shift_windows(_region(y, 0, k, k, ws, ws), nh, nw, 0, -1)              # k x shift
    br = _region(y, 0, k, 0, k, ws)                                              # k x k
    return _assemble2(tl, tr, bl, br, shift, shift, k, k)


def _zeros_like_windows(q, n):
    """n zero windows shaped like q's windows, made from q (a shared zero constant would be
    deduplicated and break the compile with "Layer names are not unique")."""
    w = q[:, :, :1]
    z = w - w
    return torch.cat([z] * n, 2) if n > 1 else z


def merge_windows(x, nh, nw):
    """Swin patch merging in window layout.

    x: plain layout of stage s, (1, C, nh*nw, 144). Returns the plain layout of stage s+1,
    (1, 4C, nh2*nw2, 144) with nh2 = ceil(nh/2), channels in HF order
    [(0,0), (1,0), (0,1), (1,1)]. Odd window grids are padded with zero windows (the next
    block masks padded tokens after its first LayerNorm)."""
    parts = []
    for dy, dx in ((0, 0), (1, 0), (0, 1), (1, 1)):
        rows = [x[:, :, :, (2 * u + dy) * WS + dx:(2 * u + dy) * WS + WS:2] for u in range(H)]
        parts.append(torch.cat(rows, 3))                         # (1, C, nW, 36)
    m = torch.cat(parts, 1)                                      # (1, 4C, nW, 36): 6x6 quadrant per old window
    nh2, nw2 = -(-nh // 2), -(-nw // 2)
    if nh % 2 or nw % 2:                                         # pad the window grid to even
        cols = []
        for b in range(nw):
            c = m[:, :, b * nh:(b + 1) * nh]
            cols.append(torch.cat([c, _zeros_like_windows(m, 1)], 2) if nh % 2 else c)
        if nw % 2:
            cols.append(_zeros_like_windows(m, 2 * nh2))
        m = torch.cat(cols, 2)
    nhe = 2 * nh2
    quads = []
    for (ra, cb) in ((0, 0), (0, 1), (1, 0), (1, 1)):             # TL, TR, BL, BR
        sel = [m[:, :, (2 * B + cb) * nhe + ra:(2 * B + cb + 1) * nhe:2] for B in range(nw2)]
        quads.append(torch.cat(sel, 2) if len(sel) > 1 else sel[0])
    return _assemble(*quads)


def unpartition(x, nh, nw, Hc=None, Wc=None):
    """Plain window layout (1, C, nh*nw, 144) -> spatial (1, C, nh*12, nw*12), cropped to Hc x Wc.
    The reshape only regroups the window and token axes (never channels)."""
    C = x.shape[1]
    cols = [x[:, :, b * nh:(b + 1) * nh, :].reshape(1, C, nh * WS, WS) for b in range(nw)]
    s = torch.cat(cols, 3)
    if Hc is not None and (Hc, Wc) != (nh * WS, nw * WS):
        s = s[:, :, :Hc, :Wc]
    return s
