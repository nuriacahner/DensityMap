# ===================================================================================================
# Mimics runner - appended to density_core.py by build_mimics_script.py (do not run this file alone).
#
# In Mimics: select one or more parts in the project panel, then Script -> Scripting Library ->
# Mapa_densidad_metodo_C. Uses the ACTIVE image (CT series). Writes, next to the project file, in the
# folder "Mapas de densidad": <proyecto>_<parte>_densidad_C.ply / _leyenda.svg / _resumen.txt.
# Nothing in the project is changed.
# ===================================================================================================
import os
import re
import time

import mimics

GV_TO_HU = -1024.0          # Mimics: grey value = HU + 1024 (Mimics help, FEA > Material Expressions)
OUT_FOLDER = 'Mapas de densidad'
TITLE = 'Mapa de densidad (método C)'
CROP_MARGIN = 3.0           # mm around the part
CHECK_VOXELS = 20


def _vec(m):
    return np.asarray(m, dtype=float).ravel()[:3]


def _safe(name):
    return re.sub(r'[^\w\-]+', '_', str(name)).strip('_') or 'parte'


def _active_image():
    for im in mimics.data.images:
        if im.active:
            return im
    return None


def _volume_around(im, lo, hi):
    """The active image cropped to the box lo..hi (+margin), as a density_core.Volume in HU, plus a
    description of the series (no patient data) and a self-check against Mimics' own grey values."""
    buf = np.asarray(im.get_voxel_buffer())          # grey values, indexed (i, j, k)
    ni, nj, nk = buf.shape

    def centre(i, j, k):
        return _vec(im.get_voxel_center((int(i), int(j), int(k))))
    o = centre(0, 0, 0)
    # voxel steps from voxels far apart: Mimics gives positions to ~7 significant digits, so a step taken
    # from neighbours is off by ~1e-5 mm and adds up to ~0.01 voxel across the image - enough to fail the
    # check at bone edges (2 of 20 voxels, first run in Mimics, 2026-11-13)
    ei = (centre(ni - 1, 0, 0) - o) / max(ni - 1, 1)
    ej = (centre(0, nj - 1, 0) - o) / max(nj - 1, 1)
    pos = np.array([centre(0, 0, k) for k in range(nk)])
    n = np.cross(ei, ej)
    n /= np.linalg.norm(n)
    sn = pos @ n
    # index box of the part
    corners = np.array([[x, y, z] for x in (lo[0] - CROP_MARGIN, hi[0] + CROP_MARGIN)
                        for y in (lo[1] - CROP_MARGIN, hi[1] + CROP_MARGIN)
                        for z in (lo[2] - CROP_MARGIN, hi[2] + CROP_MARGIN)])
    fi = (corners - o) @ ei / (ei @ ei)
    fj = (corners - o) @ ej / (ej @ ej)
    order = np.argsort(sn)
    fk = np.interp(corners @ n, sn[order], order.astype(float))
    i0, i1 = max(0, int(np.floor(fi.min())) - 1), min(ni, int(np.ceil(fi.max())) + 2)
    j0, j1 = max(0, int(np.floor(fj.min())) - 1), min(nj, int(np.ceil(fj.max())) + 2)
    k0, k1 = max(0, int(np.floor(fk.min())) - 1), min(nk, int(np.ceil(fk.max())) + 2)
    if i1 - i0 < 2 or j1 - j0 < 2 or k1 - k0 < 2:
        raise RuntimeError('La parte queda fuera de la serie de imágenes activa.')
    sub = buf[i0:i1, j0:j1, k0:k1]
    hu = sub.transpose(2, 1, 0).astype(np.float32) + GV_TO_HU      # -> [k, j, i]
    spos = np.array([centre(i0, j0, k) for k in range(k0, k1)])   # each slice's own crop corner
    reverse = bool(np.any(np.diff(spos @ n) < 0))                   # slices stored the other way round
    if reverse:
        hu, spos = hu[::-1], spos[::-1]
    vol = Volume(hu, spos[0], ei / np.linalg.norm(ei), ej / np.linalg.norm(ej),
                 (np.linalg.norm(ej), np.linalg.norm(ei)), spos)
    # self-check, two exact tests at random voxels of the crop:
    #   layout   - the buffer value is exactly Mimics' grey value there
    #   geometry - the voxel centre maps back to its own index (within 0.05 voxel)
    rng = np.random.default_rng(0)
    problems = []
    for _ in range(CHECK_VOXELS):
        i, j, k = int(rng.integers(i0, i1)), int(rng.integers(j0, j1)), int(rng.integers(k0, k1))
        c = centre(i, j, k)
        mine, theirs = int(buf[i, j, k]), int(im.get_grey_value(tuple(float(x) for x in c)))
        rel = c - vol.origin
        fi, fj = rel @ vol.u / vol.di, rel @ vol.v / vol.dj
        fk = float(np.interp(c @ vol.n, vol.sn, np.arange(len(vol.sn))))
        kk = (k1 - 1 - k) if reverse else (k - k0)
        off = max(abs(fi - (i - i0)), abs(fj - (j - j0)), abs(fk - kk))
        if mine != theirs:
            problems.append('vóxel %s: valor leído %d, Mimics %d' % ((i, j, k), mine, theirs))
        elif off > 0.05:
            problems.append('vóxel %s: posición desplazada %.3f vóxeles' % ((i, j, k), off))
    if problems:
        print('\n'.join(problems))
        raise RuntimeError('Comprobación fallida en %d de %d vóxeles (no se ha calculado nada):\n  %s' % (
            len(problems), CHECK_VOXELS, '\n  '.join(problems[:3])))
    steps = np.diff(np.sort(pos @ n))
    desc = 'serie activa del proyecto (%d x %d x %d, píxel %.2f mm, cortes cada %.2f mm)' % (
        ni, nj, nk, np.linalg.norm(ei), float(np.median(steps)) if len(steps) else 0)
    return vol, desc


def _out_dir():
    try:
        path = mimics.file.get_project_information().project_path
    except Exception:
        path = ''
    if path:
        base, proj = os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]
    else:
        base, proj = os.path.join(os.path.expanduser('~'), 'Documents'), 'proyecto_sin_guardar'
    out = os.path.join(base, OUT_FOLDER)
    os.makedirs(out, exist_ok=True)
    return out, proj


def run():
    parts = [p for p in mimics.data.parts if p.selected]
    if not parts:
        mimics.dialogs.message_box(
            'Selecciona una o más partes en el panel del proyecto (Parts) y vuelve a ejecutar el script.\n\n'
            'Se usará la serie de imágenes activa.', title=TITLE)
        return
    im = _active_image()
    if im is None:
        mimics.dialogs.message_box('No hay ninguna serie de imágenes activa en el proyecto.', title=TITLE)
        return
    names = ', '.join(p.name for p in parts)
    ans = mimics.dialogs.question_box(
        'Se calculará el mapa de densidad (método C, triángulos de ~1 mm, escala fija %d a %d HU) de:\n\n%s\n\n'
        'Tarda aproximadamente medio minuto por parte y Mimics no responde mientras tanto. '
        'No se modifica nada del proyecto.\n\n¿Continuar?' % (HU_MIN, HU_MAX, names),
        buttons='Sí;No', title=TITLE)
    # Mimics' help doesn't say what question_box returns (the button text, presumably): stop only on a clear
    # "No", so an unexpected return value can't silently swallow a "Sí"
    print('question_box answered %r' % (ans,))
    if ans is False or ans == 1 or str(ans).strip().lower() in ('no', '1', 'false'):
        return
    out, proj = _out_dir()
    done, failed = [], []
    for part in parts:
        t0 = time.time()
        try:
            nodes, tris = part.get_triangles()
            V = np.asarray(nodes, dtype=float).reshape(-1, 3)
            F = np.asarray(tris).astype(np.int64).reshape(-1, 3)
            if not is_closed(F):
                raise RuntimeError('la superficie no está cerrada (tiene agujeros o aristas sueltas); '
                                   'ciérrala (p. ej. Wrap) y vuelve a probar')
            vol, desc = _volume_around(im, V.min(0), V.max(0))
            r = density_map(V, F, vol)
            base = os.path.join(out, '%s_%s_densidad_C' % (_safe(proj), _safe(part.name)))
            write_outputs(base, part.name, r, desc,
                          extra=['Proyecto: %s' % proj,
                                 'Comprobación de lectura del TC: %d/%d vóxeles iguales a Mimics' % (CHECK_VOXELS, CHECK_VOXELS)])
            done.append('%s: mediana %.0f HU, %d triángulos, %.0f s' % (part.name, r['p'][3], len(r['F']), time.time() - t0))
        except Exception as e:
            failed.append('%s: %s' % (part.name, e))
    msg = ''
    if done:
        msg += 'Mapas creados en:\n%s\n\n%s\n\n' % (out, '\n'.join(done))
        msg += 'Para verlos en Fusion: Inspect > Color Map > Add PLY files.\nLa leyenda (.svg) se abre con el navegador.'
    if failed:
        msg += '\n\nNo se ha podido calcular:\n' + '\n'.join(failed)
    mimics.dialogs.message_box(msg.strip(), title=TITLE)


run()
