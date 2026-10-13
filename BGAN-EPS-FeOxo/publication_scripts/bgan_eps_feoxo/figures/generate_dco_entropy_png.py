from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
RUN_DIR = os.environ.get("DCO_RUN_DIR", "feoxo_implicit_all_dOC_8windows")
FIGURE_SUFFIX = os.environ.get("FIGURE_SUFFIX", "_8windows")
ROOT = HERE.parent / "log" / RUN_DIR
PROFILE = ROOT / "d_CO_entropy_profile_MECP_reference.csv"
LOOPS = ROOT / "d_CO_loop_profiles.csv"


def read_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill, anchor="mm") -> None:
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor)


def draw_polyline(draw: ImageDraw.ImageDraw, points, fill, width=1, joint="curve") -> None:
    if len(points) > 1:
        draw.line(points, fill=fill, width=width, joint=joint)


def main() -> None:
    scale = 4
    width, height = 1200 * scale, 760 * scale
    s = scale
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    colors = {
        "ink": "#1f2933",
        "muted": "#5f6b76",
        "axis": "#2f3a45",
        "grid": "#d7dde3",
        "loop": "#9aa5ae",
        "profile": "#0b6b69",
        "profile_dark": "#064b49",
        "contraction": "#f8ded9",
        "expansion": "#dff0ed",
        "mecp": "#6f45a8",
        "bars": "#6d7f8f",
    }

    title = font("arialbd.ttf", 28 * s)
    subtitle = font("arial.ttf", 16 * s)
    panel_font = font("arialbd.ttf", 19 * s)
    axis_font = font("arialbd.ttf", 17 * s)
    tick_font = font("arial.ttf", 13 * s)
    small_font = font("arial.ttf", 13 * s)
    ann_font = font("arialbd.ttf", 15 * s)
    note_font = font("arial.ttf", 14 * s)

    profile = sorted(read_csv(PROFILE), key=lambda row: row["d_CO_mid"])
    loops = sorted(read_csv(LOOPS), key=lambda row: row["d_CO_mid"])

    plot = dict(x0=112 * s, x1=(1200 - 52) * s, y0=112 * s, y1=468 * s)
    bars = dict(x0=112 * s, x1=(1200 - 52) * s, y0=574 * s, y1=684 * s)
    x_min, x_max = 1.3, 3.9
    y_min, y_max = -5.6, 3.0
    count_max = 15000
    mecp_x = 2.754

    def sx(x):
        return plot["x0"] + ((x - x_min) / (x_max - x_min)) * (plot["x1"] - plot["x0"])

    def sy(y):
        return plot["y1"] - ((y - y_min) / (y_max - y_min)) * (plot["y1"] - plot["y0"])

    def by(c):
        return bars["y1"] - (min(c, count_max) / count_max) * (bars["y1"] - bars["y0"])

    def blend_rect(box, fill, alpha):
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        rgb = tuple(int(fill[i : i + 2], 16) for i in (1, 3, 5))
        odraw.rectangle(box, fill=rgb + (int(255 * alpha),))
        img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"))

    draw.text((56 * s, 46 * s), "Carboxylate-arm entropy changes along the Fe-oxo C-O approach coordinate", font=title, fill=colors["ink"], anchor="lm")
    draw.text((56 * s, 73 * s), "BGAN-EPS on 60 implicit-solvent trajectories; entropy profile referenced to the MECP C-O window", font=subtitle, fill=colors["muted"], anchor="lm")
    draw.text((56 * s, 105 * s), "A", font=panel_font, fill=colors["ink"], anchor="lm")

    # Shaded chemical regions.
    draw.rectangle((sx(x_min), plot["y0"], sx(2.6), plot["y1"]), fill=colors["contraction"])
    draw.rectangle((sx(3.0333), plot["y0"], sx(x_max), plot["y1"]), fill=colors["expansion"])

    for t in [-5, -4, -3, -2, -1, 0, 1, 2, 3]:
        y = sy(t)
        draw.line((plot["x0"], y, plot["x1"], y), fill=colors["axis"] if t == 0 else colors["grid"], width=math.ceil((1.4 if t == 0 else 0.8) * s))

    # MECP reference.
    mx = sx(mecp_x)
    for y0 in range(int(plot["y0"]), int(plot["y1"]), 11 * s):
        draw.line((mx, y0, mx, min(y0 + 6 * s, plot["y1"])), fill=colors["mecp"], width=2 * s)
    draw.text((mx + 9 * s, plot["y0"] + 19 * s), "MECP", font=ann_font, fill=colors["mecp"], anchor="lm")
    draw.text((mx + 9 * s, plot["y0"] + 38 * s), "d(C-O) = 2.754 \u00c5", font=small_font, fill=colors["muted"], anchor="lm")

    loop_names = ["bgan1_MECP_ref", "bgan2_MECP_ref", "bgan3_MECP_ref", "bgan4_MECP_ref", "bgan5_MECP_ref"]
    for loop_name in loop_names:
        points = [(sx(row["d_CO_mid"]), sy(row[loop_name])) for row in loops]
        draw_polyline(draw, points, colors["loop"], width=2 * s)

    mean_points = [(sx(row["d_CO_mid"]), sy(row["mean_minus_T_delta_S_MECP_ref"])) for row in profile]
    draw_polyline(draw, mean_points, colors["profile"], width=5 * s)

    for row in profile:
        x = sx(row["d_CO_mid"])
        y = sy(row["mean_minus_T_delta_S_MECP_ref"])
        err = row["sem"]
        yt = sy(row["mean_minus_T_delta_S_MECP_ref"] + err)
        yb = sy(row["mean_minus_T_delta_S_MECP_ref"] - err)
        draw.line((x, yt, x, yb), fill=colors["profile"], width=2 * s)
        draw.line((x - 9 * s, yt, x + 9 * s, yt), fill=colors["profile"], width=2 * s)
        draw.line((x - 9 * s, yb, x + 9 * s, yb), fill=colors["profile"], width=2 * s)
        draw.ellipse((x - 6 * s, y - 6 * s, x + 6 * s, y + 6 * s), fill="white", outline=colors["profile_dark"], width=3 * s)

    draw.line((plot["x0"], plot["y1"], plot["x1"], plot["y1"]), fill=colors["axis"], width=2 * s)
    draw.line((plot["x0"], plot["y0"], plot["x0"], plot["y1"]), fill=colors["axis"], width=2 * s)

    for t in [1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 3.9]:
        x = sx(t)
        draw.line((x, plot["y1"], x, plot["y1"] + 7 * s), fill=colors["axis"], width=1 * s)
        text_center(draw, (x, plot["y1"] + 26 * s), f"{t:.1f}", tick_font, colors["muted"])

    for t in [-5, -4, -3, -2, -1, 0, 1, 2, 3]:
        y = sy(t)
        draw.line((plot["x0"] - 7 * s, y, plot["x0"], y), fill=colors["axis"], width=1 * s)
        draw.text((plot["x0"] - 13 * s, y), f"{t}", font=tick_font, fill=colors["muted"], anchor="rm")

    text_center(draw, ((plot["x0"] + plot["x1"]) / 2, plot["y1"] + 58 * s), "d(C60-O25) / \u00c5", axis_font, colors["ink"])
    y_label = "Relative carboxylate-arm -T\u0394S / kcal mol\u207b\u00b9"
    label_img = Image.new("RGBA", (390 * s, 34 * s), (255, 255, 255, 0))
    ldraw = ImageDraw.Draw(label_img)
    ldraw.text((195 * s, 17 * s), y_label, font=axis_font, fill=colors["ink"], anchor="mm")
    label_img = label_img.rotate(90, expand=True)
    img.paste(label_img, (14 * s, int((plot["y0"] + plot["y1"]) / 2 - label_img.height / 2)), label_img)

    draw.text((sx(1.48), sy(2.55)), "short C-O: entropic penalty", font=ann_font, fill="#7f3d35", anchor="lm")
    draw.text((sx(1.48), sy(2.18)), "rebound-like contraction", font=small_font, fill=colors["muted"], anchor="lm")
    draw.text((sx(3.12), sy(-2.62)), "larger C-O: entropy favored", font=ann_font, fill=colors["profile_dark"], anchor="lm")
    draw.text((sx(3.12), sy(-2.25)), "expanded carboxylate arm", font=small_font, fill=colors["muted"], anchor="lm")

    # Legend.
    lx = plot["x1"] - 255 * s
    draw.line((lx, plot["y0"] + 15 * s, lx + 44 * s, plot["y0"] + 15 * s), fill=colors["profile"], width=5 * s)
    draw.ellipse((lx + 22 * s - 6 * s, plot["y0"] + 15 * s - 6 * s, lx + 22 * s + 6 * s, plot["y0"] + 15 * s + 6 * s), fill="white", outline=colors["profile_dark"], width=3 * s)
    draw.text((lx + 57 * s, plot["y0"] + 20 * s), "mean \u00b1 SEM", font=small_font, fill=colors["muted"], anchor="lm")
    draw.line((lx, plot["y0"] + 40 * s, lx + 44 * s, plot["y0"] + 40 * s), fill=colors["loop"], width=2 * s)
    draw.text((lx + 57 * s, plot["y0"] + 45 * s), "individual BGAN loops", font=small_font, fill=colors["muted"], anchor="lm")

    # Panel B.
    draw.text((56 * s, 548 * s), "B", font=panel_font, fill=colors["ink"], anchor="lm")
    draw.text((bars["x0"], 548 * s), "Real trajectory support per d(C-O) window", font=axis_font, fill=colors["ink"], anchor="lm")

    for t in [0, 5000, 10000, 15000]:
        y = by(t)
        draw.line((bars["x0"], y, bars["x1"], y), fill=colors["grid"], width=1 * s)
        label = "0" if t == 0 else f"{int(t/1000)}k"
        draw.text((bars["x0"] - 13 * s, y), label, font=tick_font, fill=colors["muted"], anchor="rm")

    for row in profile:
        x0 = sx(row["d_CO_low"]) + 3 * s
        x1 = sx(row["d_CO_high"]) - 3 * s
        y = by(row["real_snapshots"])
        draw.rectangle((x0, y, x1, bars["y1"]), fill=colors["bars"])
        draw.text(((x0 + x1) / 2, y - 8 * s), f"{int(row['real_snapshots'])}", font=tick_font, fill=colors["muted"], anchor="mm")

    draw.line((bars["x0"], bars["y1"], bars["x1"], bars["y1"]), fill=colors["axis"], width=2 * s)
    draw.line((bars["x0"], bars["y0"], bars["x0"], bars["y1"]), fill=colors["axis"], width=2 * s)
    label_img = Image.new("RGBA", (90 * s, 30 * s), (255, 255, 255, 0))
    ldraw = ImageDraw.Draw(label_img)
    ldraw.text((45 * s, 15 * s), "frames", font=axis_font, fill=colors["ink"], anchor="mm")
    label_img = label_img.rotate(90, expand=True)
    img.paste(label_img, (22 * s, int((bars["y0"] + bars["y1"]) / 2 - label_img.height / 2)), label_img)

    draw.text(
        ((bars["x0"] + bars["x1"]) / 2, height - 28 * s),
        "Window bars are plotted at the same d(C-O) positions as panel A; entropy values are from 5 independent BGAN-EPS loops.",
        font=note_font,
        fill=colors["muted"],
        anchor="mm",
    )

    png = HERE / f"dCO_entropy_MECP_reference{FIGURE_SUFFIX}.png"
    tiff = HERE / f"dCO_entropy_MECP_reference{FIGURE_SUFFIX}_600dpi.tif"
    pdf = HERE / f"dCO_entropy_MECP_reference{FIGURE_SUFFIX}_raster.pdf"
    img.save(png, dpi=(600, 600))
    img.save(tiff, dpi=(600, 600), compression="tiff_lzw")
    img.save(pdf, "PDF", resolution=600.0)
    print(png)
    print(tiff)
    print(pdf)


if __name__ == "__main__":
    main()
