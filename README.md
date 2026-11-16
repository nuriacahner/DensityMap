# Density Map (DensityMap)

A tool that paints a bone's **CT density on its surface**. Each point of the bone surface
gets the mean density (in Hounsfield units, HU) of the bone beneath it, and the result is
saved as a colored mesh (PLY) with a legend.

- **In Mimics:** a script you run from **Script → Scripting Library**, on any part of any
  project, using the active CT series.
- **Outside Mimics:** the same calculation from files: a bone mesh (PLY) and the CT
  (DICOM).

The colored PLY can be shown on the bone in Fusion with the ColorMap add-in (separate
repository), which also shows its legend.

<p align="center">
  <img src="docs_densitymap/img/densitymapresult.PNG" width="800"><br>
  <em>Figure 1. Measurements Dialog Overview.</em>
</p>

## Why

Mimics' own *Vertex color* option colors a surface from the CT values found just under the
surface, at most a few pixels deep. To judge the bone quality behind a point (e.g. where a
screw would go), you need the density of the bone **below** the surface, down to the middle
of the bone. This tool computes that, and writes it in a form that other programs can show
(PLY with vertex colors).

## The method ("method C", nearest triangle)

1. **Refine the surface.** The bone's surface mesh is copied and refined to triangles of
   about 1 mm (typical edge 0.9 mm), without changing its shape. The original part is never
   modified.
2. **Sample the inside.** The interior of the bone is sampled on a regular 0.5 mm grid. The
   CT value at each point is read with trilinear interpolation.
3. **Share the bone out.** Each interior point is given to the **surface triangle nearest
   to it**. A triangle's value is the **mean HU of its points**. The bone is thus divided
   between the triangles, along the middle of the bone, and every point counts once.
4. **Color the surface.** Each vertex takes the pooled mean of its triangles. The color
   comes from a **fixed scale, −100 to 1000 HU, blue → cyan → green → yellow → red**, the
   same for every bone and every case, so a color always means the same density. Values
   outside the range take the end colors and are counted in the summary.

For each part it writes three files:
- `<project>_<part>_densidad_C.ply`: the refined surface with vertex colors. It also holds
  the HU value of each vertex, and the legend in its header;
- `…_leyenda.svg`: the legend (color bar and information); open it in a web browser;
- `…_resumen.txt`: the same information as text.

## Using it in Mimics

### Setup (once)

1. **Python:** Mimics 28's scripting needs **Python 3.13, 64-bit**. Install it from
   python.org, then set its folder in Mimics: **Help → Preferences → Scripting → interpreter
   path**.
2. **numpy:** the script needs it, in the Python that Mimics uses. Run this as
   administrator:
   ```
   "C:\Program Files\Python313\python.exe" -s -m pip install numpy
   ```
   Mimics doesn't see packages installed in your user folder: that's why the command runs
   as administrator, and `-s` makes pip install into Python's own folder.
3. **The script:** in **Preferences → Scripting → Scripting library**, choose this
   repository's **`mimics_scripts`** folder, and restart Mimics.

### Each time

1. Open the project, and make the CT series you want the **active** one.
2. **Select** one or more parts in the project panel.
3. Run **Script → Scripting Library → Mapa_densidad_metodo_C** and confirm.
4. Wait: about a minute per part. Mimics doesn't respond meanwhile.

The files are written to a folder **"Mapas de densidad"** next to the project file. Nothing
in the project is changed.

**Before calculating, the script checks itself.** At 20 random voxels it compares its
reading of the CT with Mimics' own value and position, and stops without writing anything
if they disagree. It also refuses a part whose surface isn't closed.

The script's messages are in Spanish.

## Using it outside Mimics

```
python run_from_files.py <bone.ply> <ct.dcm> <out_dir> [name]
```

- **Requirements:** Python 3 with `numpy` and `pydicom`.
- **The bone PLY** must be closed, and in the CT's patient coordinates, in mm. Mimics
  exports its parts that way.
- **The CT:** an enhanced (multi-frame) CT DICOM file.

## Install

1. Clone or copy this folder anywhere.
2. Follow **Using it in Mimics → Setup**, or install `numpy` and `pydicom` for the file
   version.

If you change `density_core.py` or `mimics_runner.py`, rebuild the Mimics script with
`python build_mimics_script.py`. Mimics lists every `.py` in its library folder as a menu
entry, so the script must be a single file: the build joins the two.

## Reliability, and known limits

**Checked:**
- **Same result by both routes.** On the same bone and CT, the map made inside Mimics and
  the map made from files agree: 99 % of the vertices are identical.
- **Stable against rounding.** Moving every coordinate by 0.0001 mm changes 90 % of the
  vertices by less than 0.02 HU.
- **Plausible.** On a radius, the dense shaft (cortex) and the less dense ends (cancellous
  bone) come out as expected.
- **CT read correctly.** The script compares its CT reading with Mimics' own at 20 voxels
  on every run.

**Not checked:** there's no test against a known answer (e.g. a synthetic bone with known
densities), and no independent implementation.

**Known limits:**
- **The bone edge lowers the values slightly.** The measurements indicate that the part's
  surface lies about half a millimetre outside the mask it was made from (not yet
  confirmed). The outermost layer it includes is the CT's blurred edge, half bone and half
  soft tissue, so its values are low.
  - In one case, the whole interior averaged 222 HU, against 240 HU for the mask in Mimics;
    without the outer 0.5 mm it was 241 HU.
  - The pattern is right. The effect is systematic, so maps made the same way compare
    fairly, but the values read slightly low where the bone is thin.
- **The scale was tuned on one case.** −100 to 1000 HU fits the averages this method gives,
  which mix cortex with marrow. A different population or CT protocol may need another
  range.
- **The CT itself limits what can be seen.** Thick slices and smooth reconstruction filters
  blur the thin cortex, which then reads lower than it is. Trabecular structure is far below
  a clinical CT's resolution.
- **HU are not calibrated bone density.** Without a calibration phantom, values compare well
  within a case or a protocol, not across scanners.
- **This is a research and visualisation aid, not certified medical software.**

## Repository layout

| Path | |
|---|---|
| `density_core.py` | the whole calculation, numpy only: refining, sampling, method C, colors, PLY and legend writing |
| `mimics_runner.py` | the Mimics side: selected parts, active series, self-check, dialogs, output folder |
| `build_mimics_script.py` | joins the two into the single Mimics script |
| `mimics_scripts/Mapa_densidad_metodo_C.py` | the built script: point Mimics' Scripting Library here |
| `run_from_files.py` | the calculation from a PLY and a DICOM file |

## Licence

MIT. See `LICENSE`.
