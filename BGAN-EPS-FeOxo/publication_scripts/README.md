# Publication-specific scripts

The reusable commands are documented in the main README. This directory also
preserves saved scripts used for the Fe(PDP) publication analyses. These use the
original system's atom map, windows, and CSV column names.

## Validation

- `validation/analyze_real_frame_entropy.py`: real-only MIST and Schlitter
  quasi-harmonic estimates, histogram sensitivity, equal-size samples, trajectory
  jackknife, and bootstrap analyses.
- `validation/summarize_entropy_validation.py`: combine the five-loop BGAN profile
  with these independent estimates and summarize pre-specified contrasts.

Supply explicit directories as shown in `docs/PUBLICATION_WORKFLOWS.md`.

## Original figure generators

The four original scripts are retained under `bgan_eps_feoxo/figures/`:

| Script | Function |
|---|---|
| `generate_dco_entropy_figure.js` | SVG of the reference-window entropy profile and loops |
| `generate_dco_entropy_png.py` | PNG/TIFF/raster PDF rendering of the entropy profile |
| `generate_dco_energy_entropy_profile.py` | Trajectory-potential energy plus entropy figure |
| `generate_dco_constrained_energy_entropy_profile.py` | Constrained-relaxed energy plus entropy figure and table |

For these scripts, place your own input tables in the following layout **inside
this directory**. The listed tables are not distributed in the code archive:

```text
publication_scripts/
  bgan_eps_feoxo/
    figures/                            # included code
    log/feoxo_implicit_all_dOC_8windows/
      d_CO_entropy_profile_MECP_reference.csv
      d_CO_loop_profiles.csv
      d_CO_energy_entropy_approx_free_energy_profile.csv
  8Window-Snapshots/
    constrained_optimization_window_energy_summary.csv
```

Run from the package root:

```bash
node publication_scripts/bgan_eps_feoxo/figures/generate_dco_entropy_figure.js
python publication_scripts/bgan_eps_feoxo/figures/generate_dco_entropy_png.py
python publication_scripts/bgan_eps_feoxo/figures/generate_dco_energy_entropy_profile.py
python publication_scripts/bgan_eps_feoxo/figures/generate_dco_constrained_energy_entropy_profile.py
```

Node.js is required only for the SVG JavaScript script; it has no additional npm
dependencies. The raster scripts require Pillow (`.[publication]`). They select
Windows fonts when present and otherwise use Pillow's default font; appearance
can vary across operating systems. The generic `bgan-eps-plot` command avoids
this legacy layout and accepts any summary produced by the reusable utility.

The optional `DCO_RUN_DIR` and `FIGURE_SUFFIX` environment variables select another
run directory/output suffix in the entropy SVG and PNG generators. The
energy/entropy scripts retain their publication-specific input names. The combined
energy/entropy curve is an approximate decomposition, not a full equilibrium PMF.
