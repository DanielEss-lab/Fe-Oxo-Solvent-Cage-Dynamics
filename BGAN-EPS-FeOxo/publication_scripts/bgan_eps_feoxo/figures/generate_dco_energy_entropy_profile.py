from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "log" / "feoxo_implicit_all_dOC_8windows" / "d_CO_energy_entropy_approx_free_energy_profile.csv"


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
        return rows


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in [Path("C:/Windows/Fonts") / name, Path("C:/Windows/Fonts/arial.ttf")]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def draw_rotated_label(img: Image.Image, text: str, center: tuple[int, int], fnt, fill: str) -> None:
    probe = Image.new("RGBA", (10, 10), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), text, font=fnt)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    label = Image.new("RGBA", (text_w + 80, text_h + 50), (255, 255, 255, 0))
    draw = ImageDraw.Draw(label)
    draw.text(((text_w + 80) / 2, (text_h + 50) / 2), text, font=fnt, fill=fill, anchor="mm")
    label = label.rotate(90, expand=True)
    img.paste(label, (int(center[0] - label.width / 2), int(center[1] - label.height / 2)), label)


def svg_path(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))


def make_svg(rows: list[dict[str, float]], out: Path) -> None:
    width, height = 1200, 760
    plot = {"x0": 112, "x1": 1138, "y0": 130, "y1": 620}
    x_min, x_max = 1.3, 3.9
    y_min, y_max = -17.5, 3.0
    mecp_x = 2.754

    def sx(x: float) -> float:
        return plot["x0"] + ((x - x_min) / (x_max - x_min)) * (plot["x1"] - plot["x0"])

    def sy(y: float) -> float:
        return plot["y1"] - ((y - y_min) / (y_max - y_min)) * (plot["y1"] - plot["y0"])

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

    def poly(key: str) -> str:
        pts = [(sx(r["mid"]), sy(r[key])) for r in rows]
        return svg_path(pts)

    svg = [f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Energy entropy and approximate free energy profile along C-O distance">
<defs>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: {colors["ink"]}; }}
    .title {{ font-size: 28px; font-weight: 700; }}
    .subtitle {{ font-size: 16px; fill: {colors["muted"]}; }}
    .axis-label {{ font-size: 18px; font-weight: 700; }}
    .tick {{ font-size: 13px; fill: {colors["muted"]}; }}
    .small {{ font-size: 13px; fill: {colors["muted"]}; }}
    .ann {{ font-size: 15px; font-weight: 700; }}
  </style>
</defs>
<rect width="{width}" height="{height}" fill="#ffffff"/>
<text x="56" y="50" class="title">Energy and carboxylate-arm entropy along the Fe-oxo C-O coordinate</text>
<text x="56" y="78" class="subtitle">All terms are referenced to the MECP-centered d(C60-O25) window; ΔGapprox = ΔE + Δ(-TΔSarm)</text>
''']

    for t in [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2]:
        svg.append(f'<line x1="{plot["x0"]}" x2="{plot["x1"]}" y1="{sy(t):.2f}" y2="{sy(t):.2f}" stroke="{colors["axis"] if t == 0 else colors["grid"]}" stroke-width="{1.4 if t == 0 else 0.8}" opacity="0.85"/>\n')

    svg.append(f'<line x1="{sx(mecp_x):.2f}" x2="{sx(mecp_x):.2f}" y1="{plot["y0"]}" y2="{plot["y1"]}" stroke="{colors["mecp"]}" stroke-width="2" stroke-dasharray="7 6"/>\n')
    svg.append(f'<text x="{sx(mecp_x)-12:.2f}" y="{plot["y0"]+22}" text-anchor="end" class="ann" fill="{colors["mecp"]}">MECP</text>\n')
    svg.append(f'<text x="{sx(mecp_x)-12:.2f}" y="{plot["y0"]+43}" text-anchor="end" class="small">reference window</text>\n')

    for key, color, width_line in [
        ("delta_E_mean_frame", colors["energy"], 3.4),
        ("minus_T_delta_S", colors["entropy"], 3.4),
        ("delta_G_approx_frame_mean", colors["free"], 4.8),
    ]:
        svg.append(f'<path d="{poly(key)}" fill="none" stroke="{color}" stroke-width="{width_line}" stroke-linecap="round" stroke-linejoin="round"/>\n')
        for row in rows:
            svg.append(f'<circle cx="{sx(row["mid"]):.2f}" cy="{sy(row[key]):.2f}" r="{5.5 if key == "delta_G_approx_frame_mean" else 4.7}" fill="#ffffff" stroke="{color}" stroke-width="2.5"/>\n')

    svg.append(f'<line x1="{plot["x0"]}" x2="{plot["x1"]}" y1="{plot["y1"]}" y2="{plot["y1"]}" stroke="{colors["axis"]}" stroke-width="1.5"/>\n')
    svg.append(f'<line x1="{plot["x0"]}" x2="{plot["x0"]}" y1="{plot["y0"]}" y2="{plot["y1"]}" stroke="{colors["axis"]}" stroke-width="1.5"/>\n')

    for t in [1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 3.9]:
        svg.append(f'<line x1="{sx(t):.2f}" x2="{sx(t):.2f}" y1="{plot["y1"]}" y2="{plot["y1"]+7}" stroke="{colors["axis"]}" stroke-width="1.2"/>\n')
        svg.append(f'<text x="{sx(t):.2f}" y="{plot["y1"]+28}" text-anchor="middle" class="tick">{t:.1f}</text>\n')

    for t in [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2]:
        svg.append(f'<line x1="{plot["x0"]-7}" x2="{plot["x0"]}" y1="{sy(t):.2f}" y2="{sy(t):.2f}" stroke="{colors["axis"]}" stroke-width="1.2"/>\n')
        svg.append(f'<text x="{plot["x0"]-13}" y="{sy(t)+4:.2f}" text-anchor="end" class="tick">{t}</text>\n')

    svg.append(f'<text x="{(plot["x0"]+plot["x1"])/2}" y="{plot["y1"]+62}" text-anchor="middle" class="axis-label">d(C60-O25) window midpoint / Å</text>\n')
    svg.append(f'<text transform="translate(52 {(plot["y0"]+plot["y1"])/2}) rotate(-90)" text-anchor="middle" class="axis-label">Relative energy term / kcal mol⁻¹</text>\n')

    legend_x, legend_y = 760, 142
    legend_items = [
        ("ΔE(dCO)", colors["energy"]),
        ("Δ[-TΔSarm(dCO)]", colors["entropy"]),
        ("ΔGapprox(dCO)", colors["free"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        y = legend_y + i * 30
        svg.append(f'<line x1="{legend_x}" x2="{legend_x+44}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="{4.8 if i == 2 else 3.4}"/>\n')
        svg.append(f'<circle cx="{legend_x+22}" cy="{y}" r="4.8" fill="#ffffff" stroke="{color}" stroke-width="2.2"/>\n')
        svg.append(f'<text x="{legend_x+58}" y="{y+5}" class="small">{label}</text>\n')

    svg.append(f'<text x="{sx(3.58):.2f}" y="{sy(-10.4):.2f}" class="ann" fill="{colors["free"]}">expanded non-rebound region</text>\n')
    svg.append(f'<text x="{sx(1.48):.2f}" y="{sy(-15.4):.2f}" class="ann" fill="{colors["free"]}">compact rebound/product region</text>\n')
    svg.append(f'<text x="{width/2}" y="{height-28}" text-anchor="middle" class="small">Potential energies are from trajectory OUT files; entropy is the 8-window BGAN-EPS arm-only profile.</text>\n')
    svg.append("</svg>\n")
    out.write_text("".join(svg), encoding="utf-8")


def make_png(rows: list[dict[str, float]], out_png: Path, out_tif: Path, out_pdf: Path) -> None:
    scale = 4
    W, H = 1200 * scale, 760 * scale
    img = Image.new("RGB", (W, H), "white")
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
    title = font("arialbd.ttf", 28 * scale)
    subtitle = font("arial.ttf", 16 * scale)
    axis_f = font("arialbd.ttf", 18 * scale)
    tick_f = font("arial.ttf", 13 * scale)
    small_f = font("arial.ttf", 13 * scale)
    ann_f = font("arialbd.ttf", 15 * scale)

    plot = {"x0": 112 * scale, "x1": 1138 * scale, "y0": 130 * scale, "y1": 620 * scale}
    x_min, x_max = 1.3, 3.9
    y_min, y_max = -17.5, 3.0
    mecp_x = 2.754

    def sx(x):
        return plot["x0"] + ((x - x_min) / (x_max - x_min)) * (plot["x1"] - plot["x0"])

    def sy(y):
        return plot["y1"] - ((y - y_min) / (y_max - y_min)) * (plot["y1"] - plot["y0"])

    def line(points, fill, width):
        draw.line(points, fill=fill, width=width, joint="curve")

    draw.text((56 * scale, 50 * scale), "Energy and carboxylate-arm entropy along the Fe-oxo C-O coordinate", font=title, fill=colors["ink"], anchor="lm")
    draw.text((56 * scale, 78 * scale), "All terms are referenced to the MECP-centered d(C60-O25) window; ΔGapprox = ΔE + Δ(-TΔSarm)", font=subtitle, fill=colors["muted"], anchor="lm")

    for t in [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2]:
        y = sy(t)
        draw.line((plot["x0"], y, plot["x1"], y), fill=colors["axis"] if t == 0 else colors["grid"], width=max(1, int((1.4 if t == 0 else 0.8) * scale)))

    mx = sx(mecp_x)
    for y in range(int(plot["y0"]), int(plot["y1"]), 13 * scale):
        draw.line((mx, y, mx, min(y + 7 * scale, plot["y1"])), fill=colors["mecp"], width=2 * scale)
    draw.text((mx - 12 * scale, plot["y0"] + 22 * scale), "MECP", font=ann_f, fill=colors["mecp"], anchor="rm")
    draw.text((mx - 12 * scale, plot["y0"] + 43 * scale), "reference window", font=small_f, fill=colors["muted"], anchor="rm")

    curves = [
        ("delta_E_mean_frame", colors["energy"], 4 * scale, "ΔE(dCO)"),
        ("minus_T_delta_S", colors["entropy"], 4 * scale, "Δ[-TΔSarm(dCO)]"),
        ("delta_G_approx_frame_mean", colors["free"], 6 * scale, "ΔGapprox(dCO)"),
    ]
    for key, color, width, _ in curves:
        pts = [(sx(row["mid"]), sy(row[key])) for row in rows]
        line(pts, color, width)
        for x, y in pts:
            r = 6 * scale if key == "delta_G_approx_frame_mean" else 5 * scale
            draw.ellipse((x - r, y - r, x + r, y + r), fill="white", outline=color, width=3 * scale)

    draw.line((plot["x0"], plot["y1"], plot["x1"], plot["y1"]), fill=colors["axis"], width=2 * scale)
    draw.line((plot["x0"], plot["y0"], plot["x0"], plot["y1"]), fill=colors["axis"], width=2 * scale)

    for t in [1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 3.9]:
        x = sx(t)
        draw.line((x, plot["y1"], x, plot["y1"] + 7 * scale), fill=colors["axis"], width=1 * scale)
        draw.text((x, plot["y1"] + 28 * scale), f"{t:.1f}", font=tick_f, fill=colors["muted"], anchor="mm")
    for t in [-16, -14, -12, -10, -8, -6, -4, -2, 0, 2]:
        y = sy(t)
        draw.line((plot["x0"] - 7 * scale, y, plot["x0"], y), fill=colors["axis"], width=1 * scale)
        draw.text((plot["x0"] - 13 * scale, y), f"{t}", font=tick_f, fill=colors["muted"], anchor="rm")

    draw.text(((plot["x0"] + plot["x1"]) / 2, plot["y1"] + 62 * scale), "d(C60-O25) window midpoint / Å", font=axis_f, fill=colors["ink"], anchor="mm")
    draw_rotated_label(img, "Relative energy term / kcal mol⁻¹", (54 * scale, int((plot["y0"] + plot["y1"]) / 2)), axis_f, colors["ink"])

    legend_x, legend_y = 760 * scale, 142 * scale
    for i, (_, color, width, label) in enumerate(curves):
        y = legend_y + i * 30 * scale
        draw.line((legend_x, y, legend_x + 44 * scale, y), fill=color, width=width)
        draw.ellipse((legend_x + 22 * scale - 5 * scale, y - 5 * scale, legend_x + 22 * scale + 5 * scale, y + 5 * scale), fill="white", outline=color, width=2 * scale)
        draw.text((legend_x + 58 * scale, y + 5 * scale), label, font=small_f, fill=colors["muted"], anchor="lm")

    draw.text((sx(3.58), sy(-10.4)), "expanded non-rebound region", font=ann_f, fill=colors["free"], anchor="mm")
    draw.text((sx(1.48), sy(-15.4)), "compact rebound/product region", font=ann_f, fill=colors["free"], anchor="lm")
    draw.text((W / 2, H - 28 * scale), "Potential energies are from trajectory OUT files; entropy is the 8-window BGAN-EPS arm-only profile.", font=small_f, fill=colors["muted"], anchor="mm")

    img.save(out_png, dpi=(600, 600))
    img.save(out_tif, dpi=(600, 600), compression="tiff_lzw")
    img.save(out_pdf, "PDF", resolution=600.0)


def main() -> None:
    rows = sorted(read_csv(DATA), key=lambda row: row["mid"])
    svg = HERE / "dCO_energy_entropy_free_energy_profile.svg"
    png = HERE / "dCO_energy_entropy_free_energy_profile.png"
    tif = HERE / "dCO_energy_entropy_free_energy_profile_600dpi.tif"
    pdf = HERE / "dCO_energy_entropy_free_energy_profile_raster.pdf"
    make_svg(rows, svg)
    make_png(rows, png, tif, pdf)
    print(svg)
    print(png)
    print(tif)
    print(pdf)


if __name__ == "__main__":
    main()
