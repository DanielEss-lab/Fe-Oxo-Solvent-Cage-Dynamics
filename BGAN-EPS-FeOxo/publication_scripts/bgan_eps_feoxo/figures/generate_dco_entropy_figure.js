const fs = require("fs");
const path = require("path");

const runDir = process.env.DCO_RUN_DIR || "feoxo_implicit_all_dOC_8windows";
const figureSuffix = process.env.FIGURE_SUFFIX || "_8windows";
const root = path.resolve(__dirname, "..", "log", runDir);
const profilePath = path.join(root, "d_CO_entropy_profile_MECP_reference.csv");
const loopPath = path.join(root, "d_CO_loop_profiles.csv");

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    const row = {};
    header.forEach((key, i) => {
      const value = values[i];
      const num = Number(value);
      row[key] = Number.isFinite(num) && value.trim() !== "" ? num : value;
    });
    return row;
  });
}

function fmt(value, digits = 1) {
  return Number(value).toFixed(digits);
}

function esc(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

const profile = parseCsv(fs.readFileSync(profilePath, "utf8")).sort((a, b) => a.d_CO_mid - b.d_CO_mid);
const loops = parseCsv(fs.readFileSync(loopPath, "utf8")).sort((a, b) => a.d_CO_mid - b.d_CO_mid);

const width = 1200;
const height = 760;
const margin = { left: 112, right: 52, top: 88, bottom: 88 };
const plot = { x0: margin.left, x1: width - margin.right, y0: 112, y1: 468 };
const bars = { x0: margin.left, x1: width - margin.right, y0: 574, y1: 684 };
const xMin = 1.3;
const xMax = 3.9;
const yMin = -5.6;
const yMax = 3.0;
const countMax = 15000;
const mecpX = 2.754;

const sx = (x) => plot.x0 + ((x - xMin) / (xMax - xMin)) * (plot.x1 - plot.x0);
const sy = (y) => plot.y1 - ((y - yMin) / (yMax - yMin)) * (plot.y1 - plot.y0);
const by = (c) => bars.y1 - (Math.min(c, countMax) / countMax) * (bars.y1 - bars.y0);

const colors = {
  ink: "#1f2933",
  muted: "#5f6b76",
  axis: "#2f3a45",
  grid: "#d7dde3",
  loop: "#9aa5ae",
  profile: "#0b6b69",
  profileDark: "#064b49",
  error: "#0b6b69",
  contraction: "#f8ded9",
  expansion: "#dff0ed",
  mecp: "#6f45a8",
  bars: "#6d7f8f",
};

function linePath(points, xAccessor, yAccessor) {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${fmt(xAccessor(p), 2)} ${fmt(yAccessor(p), 2)}`)
    .join(" ");
}

const meanPath = linePath(profile, (d) => sx(d.d_CO_mid), (d) => sy(d.mean_minus_T_delta_S_MECP_ref));
const loopNames = ["bgan1_MECP_ref", "bgan2_MECP_ref", "bgan3_MECP_ref", "bgan4_MECP_ref", "bgan5_MECP_ref"];

const ticksX = [1.3, 1.7, 2.1, 2.5, 2.9, 3.3, 3.7, 3.9];
const ticksY = [-5, -4, -3, -2, -1, 0, 1, 2, 3];
const ticksCount = [0, 5000, 10000, 15000];

let svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="MECP-referenced carboxylate-arm entropy profile along O-C distance">
<defs>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: ${colors.ink}; }
    .title { font-size: 28px; font-weight: 700; }
    .subtitle { font-size: 16px; fill: ${colors.muted}; }
    .panel { font-size: 19px; font-weight: 700; }
    .axis-label { font-size: 17px; font-weight: 600; }
    .tick { font-size: 13px; fill: ${colors.muted}; }
    .small { font-size: 13px; fill: ${colors.muted}; }
    .note { font-size: 14px; fill: ${colors.muted}; }
    .ann { font-size: 15px; font-weight: 600; }
  </style>
  <marker id="arrow-teal" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M2,2 L10,6 L2,10 Z" fill="${colors.profileDark}" />
  </marker>
  <marker id="arrow-purple" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M2,2 L10,6 L2,10 Z" fill="${colors.mecp}" />
  </marker>
</defs>
<rect width="${width}" height="${height}" fill="#ffffff"/>
<text x="56" y="46" class="title">Carboxylate-arm entropy changes along the Fe-oxo C-O approach coordinate</text>
<text x="56" y="73" class="subtitle">BGAN-EPS on 60 implicit-solvent trajectories; entropy profile referenced to the MECP C-O window</text>

<text x="56" y="105" class="panel">A</text>
<rect x="${sx(xMin)}" y="${plot.y0}" width="${sx(2.6) - sx(xMin)}" height="${plot.y1 - plot.y0}" fill="${colors.contraction}" opacity="0.55"/>
<rect x="${sx(3.0333)}" y="${plot.y0}" width="${sx(xMax) - sx(3.0333)}" height="${plot.y1 - plot.y0}" fill="${colors.expansion}" opacity="0.70"/>
`;

for (const t of ticksY) {
  svg += `<line x1="${plot.x0}" x2="${plot.x1}" y1="${sy(t)}" y2="${sy(t)}" stroke="${t === 0 ? colors.axis : colors.grid}" stroke-width="${t === 0 ? 1.4 : 0.8}" opacity="${t === 0 ? 0.8 : 0.7}"/>\n`;
}

svg += `<line x1="${sx(mecpX)}" x2="${sx(mecpX)}" y1="${plot.y0}" y2="${plot.y1}" stroke="${colors.mecp}" stroke-width="2" stroke-dasharray="6 5"/>
<text x="${sx(mecpX) + 9}" y="${plot.y0 + 19}" class="ann" fill="${colors.mecp}">MECP</text>
<text x="${sx(mecpX) + 9}" y="${plot.y0 + 38}" class="small">d(C-O) = 2.754 Å</text>
`;

for (const loopName of loopNames) {
  svg += `<path d="${linePath(loops, (d) => sx(d.d_CO_mid), (d) => sy(d[loopName]))}" fill="none" stroke="${colors.loop}" stroke-width="1.9" opacity="0.38"/>\n`;
}

svg += `<path d="${meanPath}" fill="none" stroke="${colors.profile}" stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round"/>\n`;

for (const d of profile) {
  const x = sx(d.d_CO_mid);
  const y = sy(d.mean_minus_T_delta_S_MECP_ref);
  const err = d.sem;
  const yTop = sy(d.mean_minus_T_delta_S_MECP_ref + err);
  const yBot = sy(d.mean_minus_T_delta_S_MECP_ref - err);
  svg += `<line x1="${x}" x2="${x}" y1="${yTop}" y2="${yBot}" stroke="${colors.error}" stroke-width="2.1"/>
<line x1="${x - 9}" x2="${x + 9}" y1="${yTop}" y2="${yTop}" stroke="${colors.error}" stroke-width="2.1"/>
<line x1="${x - 9}" x2="${x + 9}" y1="${yBot}" y2="${yBot}" stroke="${colors.error}" stroke-width="2.1"/>
<circle cx="${x}" cy="${y}" r="6.2" fill="#ffffff" stroke="${colors.profileDark}" stroke-width="3"/>
`;
}

svg += `<line x1="${plot.x0}" x2="${plot.x1}" y1="${plot.y1}" y2="${plot.y1}" stroke="${colors.axis}" stroke-width="1.5"/>
<line x1="${plot.x0}" x2="${plot.x0}" y1="${plot.y0}" y2="${plot.y1}" stroke="${colors.axis}" stroke-width="1.5"/>
`;

for (const t of ticksX) {
  svg += `<line x1="${sx(t)}" x2="${sx(t)}" y1="${plot.y1}" y2="${plot.y1 + 7}" stroke="${colors.axis}" stroke-width="1.2"/>
<text x="${sx(t)}" y="${plot.y1 + 26}" text-anchor="middle" class="tick">${fmt(t, 1)}</text>\n`;
}

for (const t of ticksY) {
  svg += `<line x1="${plot.x0 - 7}" x2="${plot.x0}" y1="${sy(t)}" y2="${sy(t)}" stroke="${colors.axis}" stroke-width="1.2"/>
<text x="${plot.x0 - 13}" y="${sy(t) + 4}" text-anchor="end" class="tick">${t}</text>\n`;
}

svg += `<text x="${(plot.x0 + plot.x1) / 2}" y="${plot.y1 + 58}" text-anchor="middle" class="axis-label">d(C60-O25) / Å</text>
<text transform="translate(34 ${(plot.y0 + plot.y1) / 2}) rotate(-90)" text-anchor="middle" class="axis-label">Relative carboxylate-arm -TΔS / kcal mol⁻¹</text>

<line x1="${sx(1.95)}" y1="${sy(2.25)}" x2="${sx(1.65)}" y2="${sy(1.25)}" stroke="#9c4b3f" stroke-width="2.2" marker-end="url(#arrow-purple)"/>
<text x="${sx(1.48)}" y="${sy(2.55)}" text-anchor="start" class="ann" fill="#7f3d35">short C-O: entropic penalty</text>
<text x="${sx(1.48)}" y="${sy(2.18)}" text-anchor="start" class="small">rebound-like contraction</text>

<line x1="${sx(3.15)}" y1="${sy(-3.15)}" x2="${sx(3.52)}" y2="${sy(-4.25)}" stroke="${colors.profileDark}" stroke-width="2.2" marker-end="url(#arrow-teal)"/>
<text x="${sx(3.12)}" y="${sy(-2.62)}" text-anchor="start" class="ann" fill="${colors.profileDark}">larger C-O: entropy favored</text>
<text x="${sx(3.12)}" y="${sy(-2.25)}" text-anchor="start" class="small">expanded carboxylate arm</text>

<line x1="${plot.x1 - 255}" x2="${plot.x1 - 211}" y1="${plot.y0 + 15}" y2="${plot.y0 + 15}" stroke="${colors.profile}" stroke-width="4.4"/>
<circle cx="${plot.x1 - 233}" cy="${plot.y0 + 15}" r="5.5" fill="#ffffff" stroke="${colors.profileDark}" stroke-width="2.5"/>
<text x="${plot.x1 - 198}" y="${plot.y0 + 20}" class="small">mean ± SEM</text>
<line x1="${plot.x1 - 255}" x2="${plot.x1 - 211}" y1="${plot.y0 + 40}" y2="${plot.y0 + 40}" stroke="${colors.loop}" stroke-width="1.9" opacity="0.55"/>
<text x="${plot.x1 - 198}" y="${plot.y0 + 45}" class="small">individual BGAN loops</text>

<text x="56" y="548" class="panel">B</text>
<text x="${bars.x0}" y="548" class="axis-label">Real trajectory support per d(C-O) window</text>
`;

for (const t of ticksCount) {
  svg += `<line x1="${bars.x0}" x2="${bars.x1}" y1="${by(t)}" y2="${by(t)}" stroke="${colors.grid}" stroke-width="0.8" opacity="0.7"/>
<text x="${bars.x0 - 13}" y="${by(t) + 4}" text-anchor="end" class="tick">${t / 1000}${t === 0 ? "" : "k"}</text>\n`;
}

for (const d of profile) {
  const x0 = sx(d.d_CO_low) + 3;
  const x1 = sx(d.d_CO_high) - 3;
  const y = by(d.real_snapshots);
  svg += `<rect x="${x0}" y="${y}" width="${x1 - x0}" height="${bars.y1 - y}" fill="${colors.bars}" opacity="0.88"/>
<text x="${(x0 + x1) / 2}" y="${y - 8}" text-anchor="middle" class="tick">${d.real_snapshots}</text>\n`;
}

svg += `<line x1="${bars.x0}" x2="${bars.x1}" y1="${bars.y1}" y2="${bars.y1}" stroke="${colors.axis}" stroke-width="1.5"/>
<line x1="${bars.x0}" x2="${bars.x0}" y1="${bars.y0}" y2="${bars.y1}" stroke="${colors.axis}" stroke-width="1.5"/>
<text transform="translate(42 ${(bars.y0 + bars.y1) / 2}) rotate(-90)" text-anchor="middle" class="axis-label">frames</text>
<text x="${(bars.x0 + bars.x1) / 2}" y="${height - 28}" text-anchor="middle" class="note">Window bars are plotted at the same d(C-O) positions as panel A; entropy values are from 5 independent BGAN-EPS loops.</text>
</svg>`;

const svgPath = path.join(__dirname, `dCO_entropy_MECP_reference${figureSuffix}.svg`);
const pngPath = path.join(__dirname, `dCO_entropy_MECP_reference${figureSuffix}.png`);
fs.writeFileSync(svgPath, svg);

try {
  const sharp = require(process.env.SHARP_MODULE_PATH || "sharp");
  sharp(Buffer.from(svg))
    .resize({ width: 3600 })
    .png({ compressionLevel: 9 })
    .toFile(pngPath)
    .then(() => {
      console.log(`Wrote ${svgPath}`);
      console.log(`Wrote ${pngPath}`);
    })
    .catch((err) => {
      console.error(err);
      process.exitCode = 1;
    });
} catch (err) {
  console.log(`Wrote ${svgPath}`);
  console.log("PNG export skipped because sharp is unavailable.");
}
