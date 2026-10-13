from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "bgan_eps_feoxo" / "figures"
ENERGY_CSV = ROOT / "8Window-Snapshots" / "constrained_optimization_window_energy_summary.csv"
ENTROPY_CSV = ROOT / "bgan_eps_feoxo" / "log" / "feoxo_implicit_all_dOC_8windows" / "d_CO_entropy_profile_MECP_reference.csv"
OUT_CSV = ROOT / "8Window-Snapshots" / "constrained_energy_entropy_free_energy_profile.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in [Path("C:/Windows/Fonts") / name, Path("C:/Windows/Fonts/arial.ttf")]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def text_bbox_font(text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    probe = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), text, font=fnt)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_rotated_label(img: Image.Image, text: str, center: tuple[int, int], fnt, fill: str) -> None:
    text_w, text_h = text_bbox_font(text, fnt)
    label = Image.new("RGBA", (text_w + 80, text_h + 50), (255, 255, 255, 0))
    draw = ImageDraw.Draw(label)
    draw.text(((text_w + 80) / 2, (text_h + 50) / 2), text, font=fnt, fill=fill, anchor="mm")
    label = label.rotate(90, expand=True)
    img.paste(label, (int(center[0] - label.width / 2), int(center[1] - label.height / 2)), label)


def make_combined_profile() -> list[dict[str, float | str]]:
    energy = {int(row["window"]): row for row in read_csv(ENERGY_CSV)}
    entropy = {int(row["window"]): row for row in read_csv(ENTROPY_CSV)}
    rows: list[dict[str, float | str]] = []
    for window in sorted(energy):
        e = energy[window]
        s = entropy[window]
        delta_e = as_float(e, "relative_to_window4_mean_kcalmol")
        delta_e_sem = as_float(e, "relative_to_window4_sem_kcalmol")
        delta_ts = as_float(s, "mean_minus_T_delta_S_MECP_ref")
        delta_ts_sem = as_float(s, "sem")
        delta_g = delta_e + delta_ts
        delta_g_sem = math.sqrt(delta_e_sem**2 + delta_ts_sem**2)
        rows.append(
            {
                "window": window,
                "d_CO_high": as_float(e, "d_CO_high"),
                "d_CO_low": as_float(e, "d_CO_low"),
                "d_CO_mid": as_float(e, "d_CO_mid"),
                "n_energy_structures": int(float(e["n_used_final_scf"])),
                "delta_E_constrained_kcalmol": delta_e,
                "delta_E_sem_kcalmol": delta_e_sem,
                "delta_minus_T_delta_S_arm_kcalmol": delta_ts,
                "delta_minus_T_delta_S_arm_sem_kcalmol": delta_ts_sem,
                "delta_G_approx_kcalmol": delta_g,
                "delta_G_approx_sem_kcalmol": delta_g_sem,
                "product_class_counts": e["product_class_counts"],
                "real_snapshots_in_entropy_window": int(float(s["real_snapshots"])),
            }
        )

    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_svg(rows: list[dict[str, float | str]], out: Path) -> None:
    width, height = 1220, 780
    plot = {"x0": 118, "x1": 1155, "y0": 132, "y1": 626}
    x_min, x_max = 1.3, 3.9
    y_min, y_max = -35.0, 4.0
    mecp_x = 2.754
    colors = {
        "ink": "#1f2933",
        "muted": "#62707d",
        "axis": "#2f3a45",
        "grid": "#d8dee5",
        "energy": "#2c5d8a",
        "entropy": "#0b6b69",
        "free": "#8a3d61",
        "mecp": "#6f45a8",
    }

    def sx(x: float) -> float:
        return plot["x0"] + ((x - x_min) / (x_max - x_min)) * (plot["x1"] - plot["x0"])

    def sy(y: float) -> float:
        return plot["y1"] - ((y - y_min) / (y_max - y_min)) * (plot["y1"] - plot["y0"])

    def path_for(key: str) -> str:
        return " ".join(
            f"{'M' if i == 0 else 'L'} {sx(float(row['d_CO_mid'])):.2f} {sy(float(row[key])):.2f}"
            for i, row in enumerate(rows)
        )

    svg = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Constrained energy entropy and approximate free energy profile along C-O distance">
<defs>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: {colors["ink"]}; }}
    .title {{ font-size: 27px; font-weight: 700; }}
    .subtitle {{ font-size: 15px; fill: {colors["muted"]}; }}
    .axis-label {{ font-size: 18px; font-weight: 700; }}
    .tick {{ font-size: 13px; fill: {colors["muted"]}; }}
    .small {{ font-size: 13px; fill: {colors["muted"]}; }}
    .ann {{ font-size: 15px; font-weight: 700; }}
  </style>
</defs>
<rect width="{width}" height="{height}" fill="#ffffff"/>
<text x="56" y="50" class="title">Constrained-energy and carboxylate-arm entropy profile along d(C60-O25)</text>
<text x="56" y="78" class="subtitle">ΔGapprox = ΔEconstrained + Δ(-TΔSarm); all terms referenced to the MECP-centered window</text>
''']
    for t in [-35, -30, -25, -20, -15, -10, -5, 0]:
        svg.append(f'<line x1="{plot["x0"]}" x2="{plot["x1"]}" y1="{sy(t):.2f}" y2="{sy(t):.2f}" stroke="{colors["axis"] if t == 0 else colors["grid"]}" stroke-width="{1.5 if t == 0 else 0.8}"/>\n')
    svg.append(f'<line x1="{sx(mecp_x):.2f}" x2="{sx(mecp_x):.2f}" y1="{plot["y0"]}" y2="{plot["y1"]}" stroke="{colors["mecp"]}" stroke-width="2" stroke-dasharray="7 6"/>\n')
    svg.append(f'<text x="{sx(mecp_x)-12:.2f}" y="{plot["y0"]+24}" text-anchor="end" class="ann" fill="{colors["mecp"]}">MECP</text>\n')
    svg.append(f'<text x="{sx(mecp_x)-12:.2f}" y="{plot["y0"]+44}" text-anchor="end" class="small">reference window</text>\n')

    for key, sem_key, color, stroke in [
        ("delta_E_constrained_kcalmol", "delta_E_sem_kcalmol", colors["energy"], 3.3),
        ("delta_minus_T_delta_S_arm_kcalmol", "delta_minus_T_delta_S_arm_sem_kcalmol", colors["entropy"], 3.3),
        ("delta_G_approx_kcalmol", "delta_G_approx_sem_kcalmol", colors["free"], 4.8),
    ]:
        svg.append(f'<path d="{path_for(key)}" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round"/>\n')
        for row in rows:
            x = sx(float(row["d_CO_mid"]))
            y = sy(float(row[key]))
            sem = float(row[sem_key])
            yt = sy(float(row[key]) + sem)
            yb = sy(float(row[key]) - sem)
            r = 5.6 if key == "delta_G_approx_kcalmol" else 4.8
            svg.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{yt:.2f}" y2="{yb:.2f}" stroke="{color}" stroke-width="1.6" opacity="0.75"/>\n')
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="#ffffff" stroke="{color}" stroke-width="2.4"/>\n')

    svg.append(f'<line x1="{plot["x0"]}" x2="{plot["x1"]}" y1="{plot["y1"]}" y2="{plot["y1"]}" stroke="{colors["axis"]}" stroke-width="1.5"/>\n')
    svg.append(f'<line x1="{plot["x0"]}" x2="{plot["x0"]}" y1="{plot["y0"]}" y2="{plot["y1"]}" stroke="{colors["axis"]}" stroke-width="1.5"/>\n')
    for t in [1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 3.9]:
        svg.append(f'<line x1="{sx(t):.2f}" x2="{sx(t):.2f}" y1="{plot["y1"]}" y2="{plot["y1"]+7}" stroke="{colors["axis"]}" stroke-width="1.2"/>\n')
        svg.append(f'<text x="{sx(t):.2f}" y="{plot["y1"]+28}" text-anchor="middle" class="tick">{t:.1f}</text>\n')
    for t in [-35, -30, -25, -20, -15, -10, -5, 0]:
        svg.append(f'<line x1="{plot["x0"]-7}" x2="{plot["x0"]}" y1="{sy(t):.2f}" y2="{sy(t):.2f}" stroke="{colors["axis"]}" stroke-width="1.2"/>\n')
        svg.append(f'<text x="{plot["x0"]-13}" y="{sy(t)+4:.2f}" text-anchor="end" class="tick">{t}</text>\n')
    svg.append(f'<text x="{(plot["x0"]+plot["x1"])/2}" y="{plot["y1"]+62}" text-anchor="middle" class="axis-label">d(C60-O25) window midpoint / Å</text>\n')
    svg.append(f'<text transform="translate(52 {(plot["y0"]+plot["y1"])/2}) rotate(-90)" text-anchor="middle" class="axis-label">Relative energy term / kcal mol⁻¹</text>\n')

    legend_x, legend_y = 765, 148
    legend = [
        ("ΔEconstrained", colors["energy"], 3.3),
        ("Δ(-TΔSarm)", colors["entropy"], 3.3),
        ("ΔGapprox", colors["free"], 4.8),
    ]
    for i, (label, color, stroke) in enumerate(legend):
        y = legend_y + i * 30
        svg.append(f'<line x1="{legend_x}" x2="{legend_x+44}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="{stroke}"/>\n')
        svg.append(f'<circle cx="{legend_x+22}" cy="{y}" r="4.8" fill="#ffffff" stroke="{color}" stroke-width="2.2"/>\n')
        svg.append(f'<text x="{legend_x+58}" y="{y+5}" class="small">{label}</text>\n')
    svg.append(f'<text x="{sx(3.45):.2f}" y="{sy(-30.8):.2f}" class="ann" fill="{colors["free"]}">expanded non-rebound region</text>\n')
    svg.append(f'<text x="{sx(1.48):.2f}" y="{sy(-29.5):.2f}" class="ann" fill="{colors["free"]}">compact rebound/product region</text>\n')
    svg.append(f'<text x="{width/2}" y="{height-28}" text-anchor="middle" class="small">Energies are final SCF values from constrained d(C60-O25) optimizations/post-optimization single points; entropy is the 8-window BGAN-EPS arm-only profile.</text>\n')
    svg.append("</svg>\n")
    out.write_text("".join(svg), encoding="utf-8")


def make_png(rows: list[dict[str, float | str]], out_png: Path, out_tif: Path, out_pdf: Path) -> None:
    scale = 4
    width, height = 1220 * scale, 780 * scale
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    colors = {
        "ink": "#1f2933",
        "muted": "#62707d",
        "axis": "#2f3a45",
        "grid": "#d8dee5",
        "energy": "#2c5d8a",
        "entropy": "#0b6b69",
        "free": "#8a3d61",
        "mecp": "#6f45a8",
    }
    title = font("arialbd.ttf", 27 * scale)
    subtitle = font("arial.ttf", 15 * scale)
    axis_f = font("arialbd.ttf", 18 * scale)
    tick_f = font("arial.ttf", 13 * scale)
    small_f = font("arial.ttf", 13 * scale)
    ann_f = font("arialbd.ttf", 15 * scale)
    plot = {"x0": 118 * scale, "x1": 1155 * scale, "y0": 132 * scale, "y1": 626 * scale}
    x_min, x_max = 1.3, 3.9
    y_min, y_max = -35.0, 4.0
    mecp_x = 2.754

    def sx(x):
        return plot["x0"] + ((x - x_min) / (x_max - x_min)) * (plot["x1"] - plot["x0"])

    def sy(y):
        return plot["y1"] - ((y - y_min) / (y_max - y_min)) * (plot["y1"] - plot["y0"])

    draw.text((56 * scale, 50 * scale), "Constrained-energy and carboxylate-arm entropy profile along d(C60-O25)", font=title, fill=colors["ink"], anchor="lm")
    draw.text((56 * scale, 78 * scale), "ΔGapprox = ΔEconstrained + Δ(-TΔSarm); all terms referenced to the MECP-centered window", font=subtitle, fill=colors["muted"], anchor="lm")

    for t in [-35, -30, -25, -20, -15, -10, -5, 0]:
        y = sy(t)
        draw.line((plot["x0"], y, plot["x1"], y), fill=colors["axis"] if t == 0 else colors["grid"], width=max(1, int((1.5 if t == 0 else 0.8) * scale)))
    mx = sx(mecp_x)
    for y in range(int(plot["y0"]), int(plot["y1"]), 13 * scale):
        draw.line((mx, y, mx, min(y + 7 * scale, plot["y1"])), fill=colors["mecp"], width=2 * scale)
    draw.text((mx - 12 * scale, plot["y0"] + 24 * scale), "MECP", font=ann_f, fill=colors["mecp"], anchor="rm")
    draw.text((mx - 12 * scale, plot["y0"] + 44 * scale), "reference window", font=small_f, fill=colors["muted"], anchor="rm")

    curves = [
        ("delta_E_constrained_kcalmol", "delta_E_sem_kcalmol", colors["energy"], 4 * scale, "ΔEconstrained"),
        ("delta_minus_T_delta_S_arm_kcalmol", "delta_minus_T_delta_S_arm_sem_kcalmol", colors["entropy"], 4 * scale, "Δ(-TΔSarm)"),
        ("delta_G_approx_kcalmol", "delta_G_approx_sem_kcalmol", colors["free"], 6 * scale, "ΔGapprox"),
    ]
    for key, sem_key, color, line_width, _ in curves:
        points = [(sx(float(row["d_CO_mid"])), sy(float(row[key]))) for row in rows]
        draw.line(points, fill=color, width=line_width, joint="curve")
        for row in rows:
            x = sx(float(row["d_CO_mid"]))
            y = sy(float(row[key]))
            sem = float(row[sem_key])
            yt = sy(float(row[key]) + sem)
            yb = sy(float(row[key]) - sem)
            draw.line((x, yt, x, yb), fill=color, width=max(1, 2 * scale))
            r = 6 * scale if key == "delta_G_approx_kcalmol" else 5 * scale
            draw.ellipse((x - r, y - r, x + r, y + r), fill="white", outline=color, width=3 * scale)

    draw.line((plot["x0"], plot["y1"], plot["x1"], plot["y1"]), fill=colors["axis"], width=2 * scale)
    draw.line((plot["x0"], plot["y0"], plot["x0"], plot["y1"]), fill=colors["axis"], width=2 * scale)
    for t in [1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 3.9]:
        x = sx(t)
        draw.line((x, plot["y1"], x, plot["y1"] + 7 * scale), fill=colors["axis"], width=1 * scale)
        draw.text((x, plot["y1"] + 28 * scale), f"{t:.1f}", font=tick_f, fill=colors["muted"], anchor="mm")
    for t in [-35, -30, -25, -20, -15, -10, -5, 0]:
        y = sy(t)
        draw.line((plot["x0"] - 7 * scale, y, plot["x0"], y), fill=colors["axis"], width=1 * scale)
        draw.text((plot["x0"] - 13 * scale, y), f"{t}", font=tick_f, fill=colors["muted"], anchor="rm")
    draw.text(((plot["x0"] + plot["x1"]) / 2, plot["y1"] + 62 * scale), "d(C60-O25) window midpoint / Å", font=axis_f, fill=colors["ink"], anchor="mm")
    draw_rotated_label(img, "Relative energy term / kcal mol⁻¹", (52 * scale, int((plot["y0"] + plot["y1"]) / 2)), axis_f, colors["ink"])

    legend_x, legend_y = 765 * scale, 148 * scale
    for i, (_, _, color, line_width, label) in enumerate(curves):
        y = legend_y + i * 30 * scale
        draw.line((legend_x, y, legend_x + 44 * scale, y), fill=color, width=line_width)
        draw.ellipse((legend_x + 22 * scale - 5 * scale, y - 5 * scale, legend_x + 22 * scale + 5 * scale, y + 5 * scale), fill="white", outline=color, width=2 * scale)
        draw.text((legend_x + 58 * scale, y + 5 * scale), label, font=small_f, fill=colors["muted"], anchor="lm")

    draw.text((sx(3.45), sy(-30.8)), "expanded non-rebound region", font=ann_f, fill=colors["free"], anchor="mm")
    draw.text((sx(1.48), sy(-29.5)), "compact rebound/product region", font=ann_f, fill=colors["free"], anchor="lm")
    draw.text((width / 2, height - 28 * scale), "Energies are final SCF values from constrained d(C60-O25) optimizations/post-optimization single points; entropy is the 8-window BGAN-EPS arm-only profile.", font=small_f, fill=colors["muted"], anchor="mm")
    img.save(out_png, dpi=(600, 600))
    img.save(out_tif, dpi=(600, 600), compression="tiff_lzw")
    img.save(out_pdf, "PDF", resolution=600.0)


def main() -> None:
    rows = make_combined_profile()
    svg = FIG_DIR / "dCO_constrained_energy_entropy_free_energy_profile.svg"
    png = FIG_DIR / "dCO_constrained_energy_entropy_free_energy_profile.png"
    tif = FIG_DIR / "dCO_constrained_energy_entropy_free_energy_profile_600dpi.tif"
    pdf = FIG_DIR / "dCO_constrained_energy_entropy_free_energy_profile_raster.pdf"
    make_svg(rows, svg)
    make_png(rows, png, tif, pdf)
    print(OUT_CSV)
    print(svg)
    print(png)
    print(tif)
    print(pdf)


if __name__ == "__main__":
    main()
