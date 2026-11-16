"""DensityMap core: a bone's CT density painted on its surface. numpy only, so the same code can run
inside Mimics (its Python 3.13 has numpy, nothing else).

Method C, "nearest triangle" (chosen 2026-11-13):
  every point of the bone's interior goes to the surface triangle nearest to it; a triangle's value is
  the mean CT value (HU) of its points. The bone is shared out between the triangles, split where the
  groups from opposite sides meet; every point counts once.

Pipeline (density_map()):
  refine     split every edge longer than MAX_EDGE (~1 mm triangles) - conforming, the shape is unchanged (new
             points lie on the old edges). A copy: the input mesh is not modified.
  interior   points on a regular SAMPLE_STEP grid inside the mesh (ray parity along x, row by row).
  sample     CT value at each point, trilinear (the CT is 0.71 x 0.71 mm, slices every 1.8 mm).
  assign     nearest triangle by centroid (matrix product, in chunks). Centroid instead of exact
             point-triangle distance: with 1 mm triangles the error is under ~0.5 mm.
  vertices   a vertex's value = all points of its triangles pooled (sum / count), so triangles that got
             no point still get a colour.
  colours    blue -> cyan -> green -> yellow -> red (SCALE), FIXED range HU_MIN..HU_MAX for every bone and
             case (repeatability); outside values take the end colours and are counted in the summary.
"""

import datetime
import math

import numpy as np

# mm. Splitting at midpoints overshoots: "no edge over 1.0" gave a median edge of 0.58 mm and 104k triangles on
# the radius - finer than the CT (0.71 mm pixels, 1.8 mm slices). 1.6 gives the intended ~1 mm triangles:
# median 0.92, p90 1.38 mm, 40.5k triangles (Radio, 2026-11-13).
MAX_EDGE = 1.6
SAMPLE_STEP = 0.5       # mm, interior sampling grid
HU_MIN, HU_MAX = -100.0, 1000.0

# blue -> red, the usual engineering scale (2026-11-13, the user's choice; viridis before), equally spaced
# along HU_MIN..HU_MAX, linear in between
SCALE_NAME = 'azul a rojo'
SCALE = np.array([(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)], float)


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #

def read_ply(path):
    """(V float64 (n,3), F int64 (m,3)) from a binary little-endian PLY with x y z first (Mimics)."""
    raw = open(path, 'rb').read()
    end = raw.index(b'end_header')
    body = raw.index(b'\n', end) + 1
    head = raw[:end].decode('ascii').split('\n')
    types = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'uint8': 'u1', 'char': 'i1',
             'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'short': 'i2', 'ushort': 'u2'}
    props, nv, nf, cur = [], 0, 0, None
    for line in head:
        w = line.split()
        if w[:2] == ['element', 'vertex']:
            nv, cur = int(w[2]), 'v'
        elif w[:2] == ['element', 'face']:
            nf, cur = int(w[2]), 'f'
        elif w and w[0] == 'property' and cur == 'v':
            props.append((w[2], '<' + types[w[1]]))
    vdt = np.dtype(props)
    v = np.frombuffer(raw[body:body + vdt.itemsize * nv], vdt)
    V = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    f = np.frombuffer(raw[body + vdt.itemsize * nv:], np.dtype([('n', 'u1'), ('i', '<i4', 3)]), count=nf)
    if np.any(f['n'] != 3):
        raise ValueError('only triangles are supported')
    return V, f['i'].astype(np.int64)


def write_ply(path, V, F, rgb, values=None, comment=''):
    """Binary PLY: x y z float, red green blue uchar (+ 'hu' float with the value), triangles."""
    head = ['ply', 'format binary_little_endian 1.0']
    head += ['comment ' + c for c in comment.split('\n') if c]
    head += ['element vertex %d' % len(V), 'property float x', 'property float y', 'property float z',
             'property uchar red', 'property uchar green', 'property uchar blue']
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    if values is not None:
        head.append('property float hu')
        fields.append(('hu', '<f4'))
    head += ['element face %d' % len(F), 'property list uchar int vertex_indices', 'end_header']
    v = np.zeros(len(V), np.dtype(fields))
    v['x'], v['y'], v['z'] = V[:, 0], V[:, 1], V[:, 2]
    v['r'], v['g'], v['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if values is not None:
        v['hu'] = values
    f = np.zeros(len(F), np.dtype([('n', 'u1'), ('i', '<i4', 3)]))
    f['n'], f['i'] = 3, F
    with open(path, 'wb') as fh:
        fh.write(('\n'.join(head) + '\n').encode('ascii'))
        fh.write(v.tobytes())
        fh.write(f.tobytes())


def _edges(F):
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    key = np.sort(E, 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return uniq, inv.reshape(3, -1).T            # unique edges, per-triangle edge ids (e0=ab, e1=bc, e2=ca)


def refine(V, F, max_edge=MAX_EDGE, max_rounds=12):
    """Split edges longer than max_edge until none is left. Conforming (a split edge is split in both
    of its triangles), orientation kept, new points at edge midpoints - the surface is unchanged."""
    V, F = V.copy(), F.copy()
    for _ in range(max_rounds):
        E, te = _edges(F)
        L = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        long_ = L > max_edge
        if not long_.any():
            break
        mid = np.full(len(E), -1, np.int64)
        mid[long_] = len(V) + np.arange(long_.sum())
        V = np.vstack([V, 0.5 * (V[E[long_, 0]] + V[E[long_, 1]])])
        m = mid[te]                              # (t,3) midpoint of e0/e1/e2 or -1
        k = (m >= 0).sum(1)
        out = [F[k == 0]]
        # one split edge: rotate so it is e0 (a-b); (a,m,c) (m,b,c)
        sel = k == 1
        if sel.any():
            f, mm = F[sel], m[sel]
            r = np.argmax(mm >= 0, 1)
            a, b, c = (f[np.arange(len(f)), (r + i) % 3] for i in range(3))
            m0 = mm[np.arange(len(f)), r]
            out += [np.stack([a, m0, c], 1), np.stack([m0, b, c], 1)]
        # two split edges: rotate so the unsplit one is e2 (c-a); split e0 (m0) and e1 (m1)
        sel = k == 2
        if sel.any():
            f, mm = F[sel], m[sel]
            r = (np.argmin(mm >= 0, 1) + 1) % 3   # unsplit edge index u -> start at vertex after it
            a, b, c = (f[np.arange(len(f)), (r + i) % 3] for i in range(3))
            m0 = mm[np.arange(len(f)), r]
            m1 = mm[np.arange(len(f)), (r + 1) % 3]
            out += [np.stack([m0, b, m1], 1), np.stack([a, m0, m1], 1), np.stack([a, m1, c], 1)]
        sel = k == 3
        if sel.any():
            f, mm = F[sel], m[sel]
            a, b, c = f[:, 0], f[:, 1], f[:, 2]
            m0, m1, m2 = mm[:, 0], mm[:, 1], mm[:, 2]
            out += [np.stack([a, m0, m2], 1), np.stack([b, m1, m0], 1), np.stack([c, m2, m1], 1),
                    np.stack([m0, m1, m2], 1)]
        F = np.vstack(out)
    return V, F


# --------------------------------------------------------------------------- #
# interior points
# --------------------------------------------------------------------------- #

def interior_points(V, F, step=SAMPLE_STEP):
    """Points of a regular grid (spacing `step`, slightly offset to avoid hitting vertices exactly)
    that lie inside the closed mesh. Ray parity along +x, one (y, z) row at a time per z plane."""
    lo, hi = V.min(0), V.max(0)
    off = np.array([0.0123, 0.0171, 0.0137]) * step
    xs = np.arange(lo[0] + off[0], hi[0], step)
    ys = np.arange(lo[1] + off[1], hi[1], step)
    zs = np.arange(lo[2] + off[2], hi[2], step)
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    tzmin = np.minimum(np.minimum(A[:, 2], B[:, 2]), C[:, 2])
    tzmax = np.maximum(np.maximum(A[:, 2], B[:, 2]), C[:, 2])
    pts = []
    for z in zs:
        t = np.nonzero((tzmin <= z) & (tzmax > z))[0]
        if len(t) == 0:
            continue
        a, b, c = A[t], B[t], C[t]
        # barycentric in the (y, z) projection for every row y
        d = (b[:, 1] - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (b[:, 2] - a[:, 2])
        ok = np.abs(d) > 1e-12
        a, b, c, d = a[ok], b[ok], c[ok], d[ok]
        y = ys[:, None]
        w1 = ((y - a[:, 1]) * (c[:, 2] - a[:, 2]) - (c[:, 1] - a[:, 1]) * (z - a[:, 2])) / d
        w2 = ((b[:, 1] - a[:, 1]) * (z - a[:, 2]) - (y - a[:, 1]) * (b[:, 2] - a[:, 2])) / d
        hit = (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
        xhit = a[:, 0] + w1 * (b[:, 0] - a[:, 0]) + w2 * (c[:, 0] - a[:, 0])
        for j in np.nonzero(hit.any(1))[0]:
            xc = np.sort(xhit[j][hit[j]])
            inside = (np.searchsorted(xc, xs) % 2) == 1
            if inside.any():
                xi = xs[inside]
                pts.append(np.stack([xi, np.full(len(xi), ys[j]), np.full(len(xi), z)], 1))
    return np.vstack(pts) if pts else np.zeros((0, 3))


# --------------------------------------------------------------------------- #
# CT
# --------------------------------------------------------------------------- #

class Volume:
    """A CT volume in patient coordinates (mm). hu[k, j, i]: slice k, row j, column i.
    origin = centre of voxel (0,0,0); row_dir / col_dir as in DICOM ImageOrientationPatient (row_dir is the
    direction of increasing column index i); slice_z = position of each slice along their normal."""

    def __init__(self, hu, origin, row_dir, col_dir, spacing_rc, slice_pos):
        self.hu = np.asarray(hu, np.float32)
        self.origin = np.asarray(origin, float)
        self.u = np.asarray(row_dir, float)       # +i
        self.v = np.asarray(col_dir, float)       # +j
        self.n = np.cross(self.u, self.v)
        self.di, self.dj = float(spacing_rc[1]), float(spacing_rc[0])
        s = np.asarray(slice_pos, float)          # (k, 3) positions of each slice's voxel (0,0)
        self.sn = s @ self.n
        if np.any(np.diff(self.sn) <= 0):
            raise ValueError('slices must be in increasing order along the normal')

    def sample(self, P):
        """Trilinear HU at points P (n,3); NaN outside the volume."""
        rel = P - self.origin
        fi = rel @ self.u / self.di
        fj = rel @ self.v / self.dj
        fk = np.interp(P @ self.n, self.sn, np.arange(len(self.sn)), left=-1, right=-1)
        K, J, I = self.hu.shape
        ok = (fi >= 0) & (fi <= I - 1) & (fj >= 0) & (fj <= J - 1) & (fk >= 0) & (fk <= K - 1)
        out = np.full(len(P), np.nan, np.float32)
        i0 = np.minimum(np.floor(fi[ok]).astype(int), I - 2)
        j0 = np.minimum(np.floor(fj[ok]).astype(int), J - 2)
        k0 = np.minimum(np.floor(fk[ok]).astype(int), K - 2)
        ti, tj, tk = fi[ok] - i0, fj[ok] - j0, fk[ok] - k0
        acc = 0.0
        for dk in (0, 1):
            for dj in (0, 1):
                for di in (0, 1):
                    w = (tk if dk else 1 - tk) * (tj if dj else 1 - tj) * (ti if di else 1 - ti)
                    acc = acc + w * self.hu[k0 + dk, j0 + dj, i0 + di]
        out[ok] = acc
        return out


# --------------------------------------------------------------------------- #
# method C and colours
# --------------------------------------------------------------------------- #

def nearest_triangle(P, V, F, chunk=500, candidates=8):
    """Index of the triangle whose centroid is nearest to each point.

    Fast pass in float32 (matrix products) picks `candidates` near centroids, then the exact float64
    distance decides among them. Coordinates are centred on the bone first: in patient coordinates
    (z ~ -1000 mm) the squared distances are ~1e6 and float32 couldn't separate centroids closer than
    ~0.1 mm^2, so near the surface the choice flipped with 1e-4 mm input noise - the map made inside
    Mimics differed from the one made from the PLY at 44 % of vertices (typ. 16 HU, max 883; 2026-11-13)."""
    shift = V.mean(0)
    C64 = V[F].mean(1) - shift
    P64 = P - shift
    C = C64.astype(np.float32)
    cc = (C * C).sum(1)
    out = np.empty(len(P), np.int64)
    k = min(candidates, len(C) - 1)
    for s in range(0, len(P), chunk):
        p64 = P64[s:s + chunk]
        d = cc[None, :] - 2.0 * (p64.astype(np.float32) @ C.T)   # |p|^2 is the same for the row
        cand = np.argpartition(d, k, axis=1)[:, :k + 1]
        exact = ((C64[cand] - p64[:, None, :]) ** 2).sum(2)
        out[s:s + chunk] = cand[np.arange(len(cand)), np.argmin(exact, 1)]
    return out


def vertex_values(V, F, tri, val):
    """Per-triangle sums/counts of the points, pooled at each vertex over its triangles."""
    ts = np.bincount(tri, weights=val, minlength=len(F))
    tc = np.bincount(tri, minlength=len(F)).astype(float)
    vs = np.bincount(F.ravel(), weights=np.repeat(ts, 3), minlength=len(V))
    vc = np.bincount(F.ravel(), weights=np.repeat(tc, 3), minlength=len(V))
    out = np.full(len(V), np.nan)
    has = vc > 0
    out[has] = vs[has] / vc[has]
    # vertices with no point at all: mean of neighbouring vertices, repeated
    E, _ = _edges(F)
    for _ in range(50):
        miss = np.isnan(out)
        if not miss.any():
            break
        a, b = E[:, 0], E[:, 1]
        s = np.bincount(a, weights=np.where(np.isnan(out[b]), 0, out[b]), minlength=len(V)) + \
            np.bincount(b, weights=np.where(np.isnan(out[a]), 0, out[a]), minlength=len(V))
        c = np.bincount(a, weights=~np.isnan(out[b]), minlength=len(V)) + \
            np.bincount(b, weights=~np.isnan(out[a]), minlength=len(V))
        fill = miss & (c > 0)
        out[fill] = s[fill] / c[fill]
    return out, tc


def colour_scale(values, lo=HU_MIN, hi=HU_MAX):
    t = np.clip((np.asarray(values, float) - lo) / (hi - lo), 0, 1)
    x = t * (len(SCALE) - 1)
    i = np.minimum(np.floor(x).astype(int), len(SCALE) - 2)
    f = (x - i)[:, None]
    return np.round(SCALE[i] * (1 - f) + SCALE[i + 1] * f).astype(np.uint8)


def density_map(V, F, volume, max_edge=MAX_EDGE, step=SAMPLE_STEP, log=print):
    """Method C on a copy of the mesh. Returns dict with the refined mesh, per-vertex HU and colours,
    and numbers for the summary."""
    Vr, Fr = refine(V, F, max_edge)
    log('refined: %d -> %d triangles (max edge %.1f mm)' % (len(F), len(Fr), max_edge))
    P = interior_points(V, F, step)               # interior of the original (same shape)
    log('interior points: %d (grid %.2f mm, %.1f cm3)' % (len(P), step, len(P) * step ** 3 / 1000))
    hu = volume.sample(P)
    keep = ~np.isnan(hu)
    P, hu = P[keep], hu[keep].astype(float)
    tri = nearest_triangle(P, Vr, Fr)
    vals, tcount = vertex_values(Vr, Fr, tri, hu)
    rgb = colour_scale(vals)
    below = float(np.mean(vals < HU_MIN)) * 100
    above = float(np.mean(vals > HU_MAX)) * 100
    return {'V': Vr, 'F': Fr, 'values': vals, 'rgb': rgb, 'points': len(P), 'interior_mean': float(hu.mean()),
            'tri_without_points': int((tcount == 0).sum()), 'below_pct': below, 'above_pct': above,
            'p': np.percentile(vals, [0, 5, 25, 50, 75, 95, 100])}


# --------------------------------------------------------------------------- #
# legend
# --------------------------------------------------------------------------- #

def legend_svg(path, title, lines, lo=HU_MIN, hi=HU_MAX, tick=100):
    """A vertical colour bar with HU ticks and the information lines (method, data, range...)."""
    W, H, bx, by, bw, bh = 1000, 520, 40, 60, 40, 400
    # SVG needs the stops in increasing offset: top of the bar (offset 0) is HU_MAX = the last SCALE colour
    stops = ''.join('<stop offset="%.4f" stop-color="rgb(%d,%d,%d)"/>' % (i / (len(SCALE) - 1), *c)
                    for i, c in enumerate(SCALE[::-1]))
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="Arial, Helvetica, sans-serif">' % (W, H),
         '<rect width="100%" height="100%" fill="white"/>',
         '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">%s</linearGradient></defs>' % stops,
         '<text x="%d" y="32" font-size="18" font-weight="bold">%s</text>' % (bx, _esc(title)),
         '<rect x="%d" y="%d" width="%d" height="%d" fill="url(#g)" stroke="#333"/>' % (bx, by, bw, bh)]
    v = math.ceil(lo / tick) * tick
    while v <= hi + 1e-9:
        y = by + bh * (1 - (v - lo) / (hi - lo))
        s.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#333"/>' % (bx + bw, y, bx + bw + 8, y))
        s.append('<text x="%d" y="%.1f" font-size="13" dominant-baseline="middle">%d HU</text>' % (bx + bw + 12, y, v))
        v += tick
    for i, line in enumerate(lines):
        s.append('<text x="%d" y="%d" font-size="13">%s</text>' % (bx + bw + 110, by + 12 + 21 * i, _esc(line)))
    s.append('</svg>')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(s))


def _esc(t):
    return str(t).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def legend_comment(title, lines, lo=HU_MIN, hi=HU_MAX, tick=100):
    """The legend as one PLY header comment, 'legend {json}' (ASCII: accents escaped), so the map carries
    its own legend - the ColorMap add-in in Fusion shows it in its legend panel."""
    import json
    data = {'title': title, 'unit': 'HU', 'lo': lo, 'hi': hi, 'tick': tick,
            'colors': [[int(c) for c in rgb] for rgb in SCALE], 'lines': list(lines)}
    return 'legend ' + json.dumps(data, ensure_ascii=True)


def is_closed(F):
    """Every edge used by exactly two triangles (needed for the inside test)."""
    key = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _, counts = np.unique(key, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def write_outputs(base, name, r, ct_desc, extra=()):
    """<base>.ply, <base>_leyenda.svg, <base>_resumen.txt - the same for every runner (files or Mimics).
    Returns the summary lines."""
    p = r['p']
    info = ['Hueso: %s (copia refinada a triángulos de ~1 mm: %d triángulos, arista máx. %.1f mm)' % (name, len(r['F']), MAX_EDGE),
            'Método C: cada punto del interior va al triángulo más cercano;',
            '   valor = media de la densidad (HU) de sus puntos',
            'Muestreo del interior: rejilla de %.1f mm (%d puntos), interpolación trilineal' % (SAMPLE_STEP, r['points']),
            'TC: %s' % ct_desc,
            'Escala fija: %d a %d HU (%s), igual para todos los casos' % (HU_MIN, HU_MAX, SCALE_NAME),
            'Superficie por debajo de %d HU: %.1f %%   por encima de %d HU: %.1f %%' % (HU_MIN, r['below_pct'], HU_MAX, r['above_pct']),
            'Valores en la superficie (HU): mín %.0f  P25 %.0f  mediana %.0f  P75 %.0f  máx %.0f' % (p[0], p[2], p[3], p[4], p[6]),
            'Media de todo el interior: %.0f HU' % r['interior_mean'],
            'Fecha: %s' % datetime.date.today().isoformat()] + list(extra)
    title = 'Densidad media del hueso (HU) - %s' % name
    write_ply(base + '.ply', r['V'], r['F'], r['rgb'], r['values'],
              comment='DensityMap method C (nearest triangle), HU %d..%d blue-to-red\n%s' % (
                  HU_MIN, HU_MAX, legend_comment(title, info)))
    legend_svg(base + '_leyenda.svg', title, info)
    with open(base + '_resumen.txt', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(info) + '\n')
        fh.write('Triángulos sin puntos propios (toman el valor de sus vecinos): %d\n' % r['tri_without_points'])
    return info
