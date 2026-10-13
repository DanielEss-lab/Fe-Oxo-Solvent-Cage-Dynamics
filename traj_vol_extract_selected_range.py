import subprocess, tempfile, os, csv, glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import MDAnalysis as mda

# ── Bondi (1964) for organics + Alvarez (2013) for Fe ────────────────────────
BONDI = {
    'H' : 1.20,
    'C' : 1.70,
    'N' : 1.55,
    'O' : 1.52,
    'Fe': 2.05,
}

# ── Settings ──────────────────────────────────────────────────────────────────
TRAJ_DIR      = '.'               # directory containing all XYZ trajectories
ATOM_INDICES  = [24] + list(range(51, 74))   # atoms 25 and 52–74 (1-indexed)
PROBE_RADIUS  = 1.4
MSMS_DENSITY  = 3.0
TIMESTEP_FS   = 0.75
MAX_WORKERS   = os.cpu_count()    # set lower (e.g. 8) if MSMS thrashes on I/O
OUT_VOL_CSV   = 'all_volumes_selected.csv'
OUT_AREA_CSV  = 'all_areas_selected.csv'

# ── MSMS runner ───────────────────────────────────────────────────────────────
def run_msms(coords, radii, probe_radius=PROBE_RADIUS):
    """
    Write a temporary .xyzr file, call MSMS, parse stdout.
    Returns (ses_area_A2, ses_volume_A3).
    Each call gets its own TemporaryDirectory so parallel calls never collide.
    """
    with tempfile.TemporaryDirectory() as tmp:
        xyzr = os.path.join(tmp, 'mol.xyzr')
        out  = os.path.join(tmp, 'out')

        with open(xyzr, 'w') as f:
            for (x, y, z), r in zip(coords, radii):
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {r:.4f}\n')

        cmd = ['msms',
               '-if', xyzr,
               '-probe_radius', str(probe_radius),
               '-density',      str(MSMS_DENSITY),
               '-of', out]

        result = subprocess.run(cmd, capture_output=True, text=True)

        area = vol = None

        for line in result.stdout.splitlines():
            stripped = line.strip()

            if stripped.startswith('Total ses_volume:'):
                try:
                    vol = float(stripped.split()[-1])
                except ValueError:
                    pass

            parts = stripped.split()
            if len(parts) == 4:
                try:
                    comp   = int(parts[0])
                    probe  = float(parts[1])
                    s_area = float(parts[3])
                    if comp == 0 and abs(probe - probe_radius) < 0.01:
                        area = s_area
                except ValueError:
                    pass

        if vol is None:
            raise RuntimeError(
                f'MSMS failed at probe={probe_radius}\n'
                f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
            )

        return area, vol


# ── Worker: process one frame ─────────────────────────────────────────────────
def process_frame(frame_idx, time_fs, coords, syms, traj_name):
    """Runs MSMS for a single frame. Designed to be called from a thread pool."""
    try:
        radii = np.array([BONDI[s] for s in syms])
    except KeyError as e:
        raise KeyError(
            f'Element {e} in {traj_name} frame {frame_idx} not in BONDI table.'
        )

    area_ses, vol_ses = run_msms(coords, radii)
    return frame_idx, time_fs, vol_ses, area_ses


# ── Discover all XYZ files ────────────────────────────────────────────────────
xyz_files = sorted(glob.glob(os.path.join(TRAJ_DIR, '*.xyz')))

if not xyz_files:
    raise FileNotFoundError(
        f'No .xyz files found in directory: {os.path.abspath(TRAJ_DIR)}'
    )

traj_names = [os.path.basename(f) for f in xyz_files]
print(f'Found {len(xyz_files)} trajectory file(s):')
for name in traj_names:
    print(f'  {name}')
print(f'Atom selection: atom 25 + atoms 52–74 (1-indexed), '
      f'{len(ATOM_INDICES)} atoms total\n')

# ── Process each trajectory ───────────────────────────────────────────────────
vol_data  = {}   # vol_data[frame_idx]  = {'time_fs': t, 'traj.xyz': v, ...}
area_data = {}

for xyz_path, traj_name in zip(xyz_files, traj_names):
    print(f'Processing: {traj_name}')

    u        = mda.Universe(xyz_path)
    solute   = u.atoms[ATOM_INDICES]
    n_frames = len(u.trajectory)
    n_atoms  = len(solute)

    print(f'  Frames: {n_frames}  |  Atoms selected: {n_atoms}  '
          f'|  Names (first 5): {solute.names[:5]}')

    # ── Pre-extract all frame data (MDAnalysis is not thread-safe) ────────────
    frames = []
    for ts in u.trajectory:
        frames.append((
            ts.frame,
            round(ts.frame * TIMESTEP_FS, 4),
            solute.positions.copy(),   # copy required: positions are a view
            solute.names.copy(),
        ))

    # ── Submit all frames to thread pool ──────────────────────────────────────
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for frame_idx, time_fs, coords, syms in frames:
            fut = pool.submit(process_frame, frame_idx, time_fs,
                              coords, syms, traj_name)
            futures[fut] = frame_idx

        completed = 0
        for fut in as_completed(futures):
            frame_idx, time_fs, vol_ses, area_ses = fut.result()  # re-raises on error
            completed += 1

            if frame_idx not in vol_data:
                vol_data[frame_idx]  = {'time_fs': time_fs}
                area_data[frame_idx] = {'time_fs': time_fs}

            vol_data[frame_idx][traj_name]  = round(vol_ses,  4)
            area_data[frame_idx][traj_name] = round(area_ses, 4) if area_ses is not None else ''

            print(f'  [{completed:4d}/{n_frames}] frame {frame_idx+1}  '
                  f'time={time_fs:.2f} fs  '
                  f'V_SES={vol_ses:.4f} Å³  '
                  f'SA_SES={area_ses:.4f} Å²')

    print(f'  Done: {traj_name}\n')

# ── Write CSVs ────────────────────────────────────────────────────────────────
fields        = ['time_fs'] + traj_names
sorted_frames = sorted(vol_data.keys())

with open(OUT_VOL_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows([vol_data[i] for i in sorted_frames])

with open(OUT_AREA_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows([area_data[i] for i in sorted_frames])

print(f'Volume data  → {OUT_VOL_CSV}')
print(f'Area data    → {OUT_AREA_CSV}')
print('All done.')