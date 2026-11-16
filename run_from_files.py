"""Density map from files (outside Mimics): a bone PLY exported from Mimics + the CT as DICOM.

    py -3.10 run_from_files.py <bone.ply> <ct.dcm> <out_dir> [name]

Writes <name>_densidad_C.ply (refined copy, blue-to-red vertex colours + 'hu' per vertex),
<name>_densidad_C_leyenda.svg and <name>_densidad_C_resumen.txt.
The Mimics script (to come) does the same with the part and the active image read inside Mimics.
"""

import os
import sys
import time

import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import density_core as dc


def load_dicom(path):
    ds = pydicom.dcmread(path)
    pf = ds.PerFrameFunctionalGroupsSequence
    sh = ds.SharedFunctionalGroupsSequence[0]

    def grp(name):
        return getattr(sh, name, None) or getattr(pf[0], name)
    pos = np.array([[float(x) for x in f.PlanePositionSequence[0].ImagePositionPatient] for f in pf])
    iop = [float(v) for v in grp('PlaneOrientationSequence')[0].ImageOrientationPatient]
    px = [float(v) for v in grp('PixelMeasuresSequence')[0].PixelSpacing]
    rv = grp('PixelValueTransformationSequence')[0]
    hu = ds.pixel_array.astype(np.float32) * float(rv.RescaleSlope) + float(rv.RescaleIntercept)
    n = np.cross(iop[:3], iop[3:])
    order = np.argsort(pos @ n)
    return dc.Volume(hu[order], pos[order][0], iop[:3], iop[3:], px, pos[order])


def main(ply, dcm, out_dir, name=None):
    name = name or os.path.splitext(os.path.basename(ply))[0]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    V, F = dc.read_ply(ply)
    vol = load_dicom(dcm)
    print('loaded in %.1f s: %d triangles, CT %s' % (time.time() - t0, len(F), vol.hu.shape))
    r = dc.density_map(V, F, vol)
    print('computed in %.1f s' % (time.time() - t0))
    base = os.path.join(out_dir, '%s_densidad_C' % name)
    info = dc.write_outputs(base, name, r, os.path.basename(dcm))
    print('\n'.join(info))
    print('triangles without own points:', r['tri_without_points'])
    print('written:', base + '.ply / _leyenda.svg / _resumen.txt')


if __name__ == '__main__':
    main(*sys.argv[1:])
