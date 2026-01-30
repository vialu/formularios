import os
import io
import json
import zipfile
import re
from datetime import date, datetime
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---- Text anchor extraction ----
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False


# ============================
# Paths
# ============================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
SIG_DIR = os.path.join(DATA_DIR, "signatures")
OUT_DIR = os.path.join(DATA_DIR, "outputs")
PROFILE_PATH = os.path.join(DATA_DIR, "profile.json")
TEMPLATE_STORE = os.path.join(APP_DIR, "template_store")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TEMPLATE_STORE, exist_ok=True)

USER_ID = "default_user"
SIG_PATH = os.path.join(SIG_DIR, f"{USER_ID}.png")


# ============================
# Marker set (Opción B: pares L/R + uno para sello/firma)
# ============================
DEFAULT_MARKERS = [
    "<<pn_l>>", "<<pn_r>>",      # paciente nombre
    "<<pr_l>>", "<<pr_r>>",      # paciente RUT
    "<<dg_l>>", "<<dg_r>>",      # diagnóstico
    "<<tr_l>>", "<<tr_r>>",      # tratamiento
    "<<fa_l>>", "<<fa_r>>",      # fecha atención
    "<<fd_l>>", "<<fd_r>>",      # fecha diagnóstico
    "<<mdn_l>>", "<<mdn_r>>",    # médico nombre
    "<<mde_l>>", "<<mde_r>>",    # médico especialidad
    "<<mdr_l>>", "<<mdr_r>>",    # médico RUT
    "<<sf>>",                    # sello/firma (timbre+firma traslapados)
]


# ============================
# Default bindings
# ============================
DEFAULT_BINDINGS = {
    "paciente_nombre": {
        "type": "lr_singleline",
        "left": "<<pn_l>>",
        "right": "<<pn_r>>",
        "source": "caso.paciente_nombre",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },
    "paciente_rut": {
        "type": "lr_singleline",
        "left": "<<pr_l>>",
        "right": "<<pr_r>>",
        "source": "caso.paciente_rut",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },
    "diagnostico": {
        "type": "lr_singleline",
        "left": "<<dg_l>>",
        "right": "<<dg_r>>",
        "source": "caso.diagnostico",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },
    "tratamiento": {
        "type": "lr_singleline",
        "left": "<<tr_l>>",
        "right": "<<tr_r>>",
        "source": "caso.tratamiento",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },
    "fecha_atencion": {
        "type": "lr_singleline",
        "left": "<<fa_l>>",
        "right": "<<fa_r>>",
        "source": "caso.fecha_atencion",
        "format": "date_ddmmyyyy",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },
    "fecha_diagnostico": {
        "type": "lr_singleline",
        "left": "<<fd_l>>",
        "right": "<<fd_r>>",
        "source": "caso.fecha_diagnostico",
        "format": "date_ddmmyyyy",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },

    # Médico (auto desde perfil)
    "medico_nombre": {
        "type": "lr_singleline",
        "left": "<<mdn_l>>",
        "right": "<<mdn_r>>",
        "source": "perfil.nombre",
        "prefix": "Dr. ",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },
    "medico_especialidad": {
        "type": "lr_singleline",
        "left": "<<mde_l>>",
        "right": "<<mde_r>>",
        "source": "perfil.especialidad",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },
    "medico_rut": {
        "type": "lr_singleline",
        "left": "<<mdr_l>>",
        "right": "<<mdr_r>>",
        "source": "perfil.rut",
        "max_size": 10,
        "min_size": 7,
        "align": "left",
        "pad_x": 1.0,
        "height": 12.0,
    },

    "sello_firma": {
        "type": "stamp_sig",
        "marker": "<<sf>>",
        "box_w": 200,
        "box_h": 80,
    }
}

# ============================
# Default Dynamic UI schema (compat)
# ============================
DEFAULT_UI_FIELDS = [
    {"key": "paciente_nombre", "label": "Nombre paciente", "type": "text", "required": True},
    {"key": "paciente_rut", "label": "RUT paciente", "type": "text", "required": True},
    {"key": "fecha_atencion", "label": "Fecha atención", "type": "date", "required": True},
    {"key": "fecha_diagnostico", "label": "Fecha diagnóstico", "type": "date", "required": True},
    {"key": "diagnostico", "label": "Diagnóstico (1 línea)", "type": "text", "required": True},
    {"key": "tratamiento", "label": "Tratamiento (1 línea)", "type": "text", "required": False},
]


# ============================
# Template store helpers
# ============================
def template_dir(template_id: str) -> str:
    return os.path.join(TEMPLATE_STORE, template_id)

def template_json_path(template_id: str) -> str:
    return os.path.join(template_dir(template_id), "template.json")

def template_pdf_path(template_id: str) -> str:
    return os.path.join(template_dir(template_id), "form.pdf")  # base limpio

def template_pdf_marked_path(template_id: str) -> str:
    return os.path.join(template_dir(template_id), "form_marked.pdf")  # con marcadores

def sanitize_template_id(s: str) -> str:
    s = (s or "").strip().lower()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    out = []
    for ch in s:
        out.append(ch if ch in allowed else "_")
    tid = "".join(out)
    while "__" in tid:
        tid = tid.replace("__", "_")
    return tid.strip("_")

def list_installed_templates() -> List[str]:
    out = []
    for name in os.listdir(TEMPLATE_STORE):
        d = os.path.join(TEMPLATE_STORE, name)
        if not os.path.isdir(d):
            continue
        if os.path.exists(template_json_path(name)) and os.path.exists(template_pdf_path(name)):
            out.append(name)
    return sorted(out)

def load_template(template_id: str) -> Dict[str, Any]:
    with open(template_json_path(template_id), "r", encoding="utf-8") as f:
        return json.load(f)

def save_template(template_id: str, tpl: Dict[str, Any]) -> None:
    os.makedirs(template_dir(template_id), exist_ok=True)
    with open(template_json_path(template_id), "w", encoding="utf-8") as f:
        json.dump(tpl, f, ensure_ascii=False, indent=2)


# ============================
# Profile
# ============================
def load_profile() -> Dict[str, Any]:
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            try:
                p = json.load(f)
            except Exception:
                p = {}
    else:
        p = {}

    # Backward-compatible defaults
    p.setdefault("nombre", "")
    p.setdefault("rut", "")
    p.setdefault("especialidad", "")
    p.setdefault("icm", "")
    p.setdefault("institucion", "")
    p.setdefault("correo", "")
    p.setdefault("telefono", "")
    p.setdefault("stamp_font_size", 9)
    return p

    return {"nombre": "", "rut": "", "especialidad": "", "icm": "", "stamp_font_size": 9}

def save_profile(profile: Dict[str, Any]) -> None:
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


# ============================
# Signature helpers
# ============================
def trim_whitespace(img_rgba: Image.Image, padding: int = 10) -> Image.Image:
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")
    alpha = img_rgba.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        x0 = max(0, bbox[0] - padding)
        y0 = max(0, bbox[1] - padding)
        x1 = min(img_rgba.width, bbox[2] + padding)
        y1 = min(img_rgba.height, bbox[3] + padding)
        return img_rgba.crop((x0, y0, x1, y1))
    return img_rgba

def ensure_transparent_background(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    gray = ImageOps.grayscale(img)
    threshold = 245
    alpha = gray.point(lambda p: 0 if p > threshold else 255)
    r, g, b, _ = img.split()
    return Image.merge("RGBA", (r, g, b, alpha))

def save_signature_png(img_rgba: Image.Image, path: str, max_width: int = 1200) -> None:
    img_rgba = trim_whitespace(img_rgba, padding=10)
    if img_rgba.width > max_width:
        ratio = max_width / img_rgba.width
        new_size = (int(img_rgba.width * ratio), int(img_rgba.height * ratio))
        img_rgba = img_rgba.resize(new_size, Image.LANCZOS)
    img_rgba.save(path, format="PNG")


# ============================
# Marker detection (PyMuPDF)
# ============================
def find_markers_in_pdf(pdf_bytes: bytes, markers: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not HAS_FITZ:
        raise RuntimeError("PyMuPDF (pymupdf) no está instalado.")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in markers}
    for i, page in enumerate(doc):
        for m in markers:
            rects = page.search_for(m, quads=False)
            for r in rects:
                results[m].append({
                    "page_index": i,
                    "x0": float(r.x0),
                    "y0": float(r.y0),
                    "x1": float(r.x1),
                    "y1": float(r.y1),
                })
    doc.close()
    return results


# ============================
# Drawing helpers
# ============================
def ddmmyyyy(d: date) -> str:
    return f"{d.day:02d}/{d.month:02d}/{d.year:04d}"




def calc_age_years_months(birth: Any, ref: Optional[date] = None) -> str:
    """Return age as 'Xa Ym' given a birth date. Accepts date or ISO/dd/mm/yyyy strings."""
    if ref is None:
        ref = date.today()

    b = None
    if isinstance(birth, date):
        b = birth
    elif isinstance(birth, datetime):
        b = birth.date()
    elif isinstance(birth, str):
        s = birth.strip()
        if not s:
            return ""
        try:
            b = date.fromisoformat(s)
        except Exception:
            m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
            if m:
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    b = date(y, mo, d)
                except Exception:
                    b = None

    if b is None or b > ref:
        return ""

    years = ref.year - b.year
    months = ref.month - b.month

    # If we haven't reached the birth day in the current month, subtract one month
    if ref.day < b.day:
        months -= 1

    if months < 0:
        years -= 1
        months += 12

    years = max(0, years)
    months = max(0, months)
    return f"{years}a {months}m"


def apply_transform(bind: Dict[str, Any], raw: Any, caso: Dict[str, Any]) -> Any:
    """Apply optional transforms declared in binding."""
    tname = bind.get("transform")
    if not tname:
        return raw

    tname = str(tname).strip().lower()

    if tname == "age_years_months":
        ref_path = bind.get("transform_ref")
        ref_date = None
        if isinstance(ref_path, str) and ref_path.startswith("caso."):
            ref_key = ref_path.split(".", 1)[1]
            ref_date = caso.get(ref_key, None)
            if isinstance(ref_date, datetime):
                ref_date = ref_date.date()
            if not isinstance(ref_date, date):
                ref_date = None
        return calc_age_years_months(raw, ref=ref_date)

    return raw


def parse_date_default(default: Any) -> date:
    """Parse default date values coming from JSON/UI.
    - None or 'today' -> today
    - date/datetime -> date
    - 'YYYY-MM-DD' or 'DD/MM/YYYY' -> parsed date
    Falls back to today if parsing fails.
    """
    if default is None:
        return date.today()
    if isinstance(default, datetime):
        return default.date()
    if isinstance(default, date):
        return default
    if isinstance(default, str):
        s = default.strip()
        if not s:
            return date.today()
        if s.lower() == "today":
            return date.today()
        # ISO
        try:
            return date.fromisoformat(s)
        except Exception:
            pass
        # dd/mm/yyyy
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return date(y, mo, d)
            except Exception:
                return date.today()
    return date.today()

def resolve_value(path: str, caso: Dict[str, Any], perfil: Dict[str, Any]) -> Any:
    if path.startswith("caso."):
        k = path.split(".", 1)[1]
        return caso.get(k, "")
    if path.startswith("perfil."):
        k = path.split(".", 1)[1]
        return perfil.get(k, "")
    return ""

def format_value(v: Any, fmt: Optional[str]) -> str:
    if fmt == "date_ddmmyyyy":
        if isinstance(v, date):
            return ddmmyyyy(v)
        return ""
    return "" if v is None else str(v)

def draw_singleline_autoshrink(c: canvas.Canvas,
                               text: str,
                               box_x0: float, box_y0: float,
                               box_w: float, box_h: float,
                               font: str,
                               max_size: int,
                               min_size: int,
                               align: str = "left",
                               pad_x: float = 1.0) -> None:
    text = "" if text is None else str(text)

    avail_w = max(1.0, box_w - 2 * pad_x)
    size = int(max_size)
    while size > int(min_size):
        c.setFont(font, size)
        if c.stringWidth(text, font, size) <= avail_w:
            break
        size -= 1
    c.setFont(font, size)

    ty = box_y0 + (box_h / 2) - (size * 0.35)

    if align == "center":
        c.drawCentredString(box_x0 + box_w / 2, ty, text)
    elif align == "right":
        c.drawRightString(box_x0 + box_w - pad_x, ty, text)
    else:
        c.drawString(box_x0 + pad_x, ty, text)

def top_to_bottom_coords(ph: float, hit: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x0 = float(hit["x0"])
    x1 = float(hit["x1"])
    y0_top = float(hit["y0"])
    y1_top = float(hit["y1"])
    y1_bl = ph - y0_top
    y0_bl = ph - y1_top
    return x0, y0_bl, x1, y1_bl

def pick_first_hit(hits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return hits[0] if hits else None


# ============================
# Overlay generation using L/R pairs
# ============================
def make_overlay(reader: PdfReader,
                 template: Dict[str, Any],
                 caso: Dict[str, Any],
                 perfil: Dict[str, Any],
                 signature_png_path: str) -> PdfReader:

    anchors = template.get("anchors", {})
    bindings = template.get("bindings", DEFAULT_BINDINGS)
    style = template.get("style", {})

    needs_cover = bool(template.get("needs_cover", False))
    cover_padding = float(style.get("cover_padding", 1.5))

    default_font = style.get("font", "Helvetica")
    bold_font = style.get("font_bold", "Helvetica-Bold")

    stamp_font_size = int(perfil.get("stamp_font_size", style.get("stamp_font_size", 9)))
    stamp_y_ratio = float(style.get("stamp_y_ratio", 0.22))
    sig_h_ratio = float(style.get("sig_h_ratio", 0.85))
    stamp_gap = float(style.get("stamp_gap", 1.0))

    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=(612, 792))

    for page_index, page in enumerate(reader.pages):
        pw = float(page.mediabox.width)
        ph = float(page.mediabox.height)
        c.setPageSize((pw, ph))

        for field_key, bind in bindings.items():
            btype = bind.get("type", "")

            if btype in ("lr_singleline", "lr_multiline"):
                lm = bind.get("left")
                rm = bind.get("right")
                if not lm or not rm:
                    continue

                lh = pick_first_hit([h for h in anchors.get(lm, []) if int(h.get("page_index", -1)) == page_index])
                rh = pick_first_hit([h for h in anchors.get(rm, []) if int(h.get("page_index", -1)) == page_index])
                if not lh or not rh:
                    continue

                lx0, ly0, lx1, ly1 = top_to_bottom_coords(ph, lh)
                rx0, ry0, rx1, ry1 = top_to_bottom_coords(ph, rh)

                box_x0 = lx1
                box_x1 = rx0
                box_w = max(5.0, box_x1 - box_x0)

                center_y = (ly0 + ly1) / 2
                box_h = float(bind.get("height", 12.0))
                box_y0 = center_y - box_h / 2

                dy = float(bind.get("dy", 0.0))
                box_y0 += dy

                if needs_cover:
                    for hit in (lh, rh):
                        mx0, my0, mx1, my1 = top_to_bottom_coords(ph, hit)
                        c.saveState()
                        c.setFillColorRGB(1, 1, 1)
                        c.rect(mx0 - cover_padding, my0 - cover_padding,
                               (mx1 - mx0) + 2 * cover_padding,
                               (my1 - my0) + 2 * cover_padding,
                               stroke=0, fill=1)
                        c.restoreState()

                src = bind.get("source", "")
                raw = resolve_value(src, caso, perfil)

                # Optional transform (e.g., age from birthdate)
                raw = apply_transform(bind, raw, caso)

                fmt = bind.get("format")
                text = format_value(raw, fmt)

                prefix = bind.get("prefix", "")
                if prefix and text:
                    text = f"{prefix}{text}"
                elif prefix and not text and src.startswith("perfil."):
                    # si es perfil, igual mostrar prefix aunque venga vacío? normalmente no
                    pass

                font = bind.get("font", default_font)
                align = str(bind.get("align", "left")).lower()
                pad_x = float(bind.get("pad_x", 1.0))

                if btype == "lr_singleline":
                    max_size = int(bind.get("max_size", 10))
                    min_size = int(bind.get("min_size", 7))
                    draw_singleline_autoshrink(
                        c, text,
                        box_x0, box_y0,
                        box_w, box_h,
                        font=font,
                        max_size=max_size,
                        min_size=min_size,
                        align=align,
                        pad_x=pad_x
                    )
                else:
                    # Minimal lr_multiline (si lo usas a futuro)
                    size = int(bind.get("size", 10))
                    line_height = float(bind.get("line_height", size + 2))
                    c.setFont(font, size)
                    words = text.replace("\n", " \n ").split(" ")
                    lines = []
                    cur = ""
                    for w in words:
                        if w == "\n":
                            lines.append(cur.rstrip())
                            cur = ""
                            continue
                        trial = (cur + " " + w).strip() if cur else w
                        if c.stringWidth(trial, font, size) > (box_w - 2 * pad_x) and cur:
                            lines.append(cur.rstrip())
                            cur = w
                        else:
                            cur = trial
                    if cur:
                        lines.append(cur.rstrip())

                    yy = box_y0 + box_h - size
                    for ln in lines:
                        if yy < box_y0:
                            break
                        c.drawString(box_x0 + pad_x, yy, ln)
                        yy -= line_height

            elif btype == "stamp_sig":
                marker = bind.get("marker", "<<sf>>")
                hit = pick_first_hit([h for h in anchors.get(marker, []) if int(h.get("page_index", -1)) == page_index])
                if not hit:
                    continue

                mx0, my0, mx1, my1 = top_to_bottom_coords(ph, hit)
                mb_w = (mx1 - mx0)
                mb_h = (my1 - my0)
                cx = mx0 + mb_w / 2
                cy = my0 + mb_h / 2

                box_w = float(bind.get("box_w", 200))
                box_h = float(bind.get("box_h", 80))
                box_x0 = cx - box_w / 2
                box_y0 = cy - box_h / 2

                if needs_cover:
                    c.saveState()
                    c.setFillColorRGB(1, 1, 1)
                    c.rect(mx0 - cover_padding, my0 - cover_padding,
                           mb_w + 2 * cover_padding, mb_h + 2 * cover_padding,
                           stroke=0, fill=1)
                    c.restoreState()

                stamp_lines = [
                    f"Dr. {perfil.get('nombre','')}".strip(),
                    f"{perfil.get('especialidad','')}".strip(),
                    f"RUT: {perfil.get('rut','')}".strip(),
                ]

                # Draw stamp first
                c.setFont(bold_font, stamp_font_size)
                line_gap = stamp_font_size + stamp_gap
                stamp_block_h = line_gap * len(stamp_lines)

                stamp_center_y = box_y0 + box_h * stamp_y_ratio + stamp_block_h / 2
                start_y = stamp_center_y + (len(stamp_lines) - 1) * line_gap / 2
                for line in stamp_lines:
                    c.drawCentredString(cx, start_y, line)
                    start_y -= line_gap

                # Signature over it
                if signature_png_path and os.path.exists(signature_png_path):
                    img = ImageReader(signature_png_path)
                    iw, ih = img.getSize()

                    sig_area_h = box_h * sig_h_ratio
                    sig_area_y = box_y0 + (box_h - sig_area_h) / 2

                    pad = 2.0
                    aw, ah = (box_w - 2 * pad), (sig_area_h - 2 * pad)
                    scale = min(aw / iw, ah / ih) if iw and ih else 1.0
                    dw, dh = iw * scale, ih * scale

                    dx = box_x0 + (box_w - dw) / 2
                    dy = sig_area_y + (sig_area_h - dh) / 2
                    c.drawImage(img, dx, dy, width=dw, height=dh, mask="auto")

            elif btype == "checkbox":
                marker = bind.get("marker")
                if not marker:
                    continue
                hit = pick_first_hit([h for h in anchors.get(marker, []) if int(h.get("page_index", -1)) == page_index])
                if not hit:
                    continue

                raw = resolve_value(bind.get("source", ""), caso, perfil)

                # Interpretación robusta de True/False (soporta strings tipo "si", "x", "true", etc.)
                if isinstance(raw, bool):
                    checked = raw
                else:
                    s = str(raw).strip().lower()
                    checked = s in ("1", "true", "t", "yes", "y", "si", "sí", "x", "check", "checked", "on")

                mx0, my0, mx1, my1 = top_to_bottom_coords(ph, hit)

                if needs_cover:
                    c.saveState()
                    c.setFillColorRGB(1, 1, 1)
                    c.rect(mx0 - cover_padding, my0 - cover_padding,
                           (mx1 - mx0) + 2 * cover_padding,
                           (my1 - my0) + 2 * cover_padding,
                           stroke=0, fill=1)
                    c.restoreState()

                if checked:
                    glyph = bind.get("glyph", "X")
                    size = int(bind.get("size", 11))
                    font = bind.get("font", bold_font)

                    cx = (mx0 + mx1) / 2
                    cy = (my0 + my1) / 2

                    c.setFont(font, size)
                    ty = cy - (size * 0.35)
                    c.drawCentredString(cx, ty, glyph)

        c.showPage()

    c.save()
    packet.seek(0)
    return PdfReader(packet)


def merge_overlay(base_pdf_bytes: bytes, overlay_reader: PdfReader) -> bytes:
    base_reader = PdfReader(io.BytesIO(base_pdf_bytes))
    writer = PdfWriter()
    for p in base_reader.pages:
        writer.add_page(p)
    for i in range(len(writer.pages)):
        writer.pages[i].merge_page(overlay_reader.pages[i])
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()


# ============================
# Template builder
# ============================
def build_template(template_id: str,
                   pdf_marked_bytes: bytes,
                   pdf_base_bytes: Optional[bytes],
                   name: str,
                   vendor: str,
                   markers: List[str]) -> Dict[str, Any]:
    hits = find_markers_in_pdf(pdf_marked_bytes, markers)
    anchors = {m: v for m, v in hits.items() if v}
    needs_cover = pdf_base_bytes is None

    tpl = {
        "template_id": template_id,
        "name": name or template_id,
        "vendor": vendor or "",
        "pdf": {"filename": "form.pdf", "marked_filename": "form_marked.pdf"},
        "markers": markers,
        "anchors": anchors,
        "bindings": DEFAULT_BINDINGS,
        "needs_cover": needs_cover,
        "style": {
            "font": "Helvetica",
            "font_bold": "Helvetica-Bold",
            "cover_padding": 1.5,
            "stamp_font_size": 9,
            "stamp_y_ratio": 0.22,
            "sig_h_ratio": 0.85,
            "stamp_gap": 1.0,
        },
        # NEW: dynamic UI schema (safe default)
        "ui": {
            "fields": DEFAULT_UI_FIELDS
        },
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(template_dir(template_id), exist_ok=True)
    with open(template_pdf_marked_path(template_id), "wb") as f:
        f.write(pdf_marked_bytes)
    if pdf_base_bytes is None:
        pdf_base_bytes = pdf_marked_bytes
    with open(template_pdf_path(template_id), "wb") as f:
        f.write(pdf_base_bytes)

    save_template(template_id, tpl)
    return tpl


# ============================
# Zip import/export
# ============================
def export_template_zip_bytes(template_id: str) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(template_json_path(template_id), arcname=f"{template_id}/template.json")
        z.write(template_pdf_path(template_id), arcname=f"{template_id}/form.pdf")
        marked = template_pdf_marked_path(template_id)
        if os.path.exists(marked):
            z.write(marked, arcname=f"{template_id}/form_marked.pdf")
    mem.seek(0)
    return mem.read()

def import_template_zip(uploaded_bytes: bytes) -> List[str]:
    installed = []
    mem = io.BytesIO(uploaded_bytes)
    with zipfile.ZipFile(mem, "r") as z:
        roots = set()
        for name in z.namelist():
            if "/" in name:
                roots.add(name.split("/", 1)[0])
        for rid in roots:
            tj = f"{rid}/template.json"
            fp = f"{rid}/form.pdf"
            mp = f"{rid}/form_marked.pdf"
            if tj in z.namelist() and fp in z.namelist():
                os.makedirs(template_dir(rid), exist_ok=True)
                with open(template_json_path(rid), "wb") as f:
                    f.write(z.read(tj))
                with open(template_pdf_path(rid), "wb") as f:
                    f.write(z.read(fp))
                if mp in z.namelist():
                    with open(template_pdf_marked_path(rid), "wb") as f:
                        f.write(z.read(mp))
                installed.append(rid)
    return installed


# ============================
# UI helpers
# ============================
def profile_signature_ui():
    st.subheader("Firma")
    tab1, tab2 = st.tabs(["Dibujar firma (recomendado)", "Subir imagen (opcional)"])

    with tab1:
        st.caption("Dibuja tu firma con mouse/trackpad. Luego presiona **Guardar firma**.")
        canvas_res = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#000000",
            background_color="rgba(255, 255, 255, 0)",
            height=220,
            width=700,
            drawing_mode="freedraw",
            key="sig_canvas",
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Guardar firma (canvas)"):
                if canvas_res.image_data is None:
                    st.error("No se detectó firma.")
                else:
                    img = Image.fromarray(canvas_res.image_data.astype("uint8"), mode="RGBA")
                    save_signature_png(img, SIG_PATH)
                    st.success("Firma guardada.")
        with c2:
            if st.button("Borrar y volver a dibujar"):
                st.session_state["sig_canvas"] = None
                st.rerun()

    with tab2:
        st.caption("Sube una firma escaneada o foto (ideal: fondo blanco). La app la limpiará.")
        up = st.file_uploader("Archivo de firma", type=["png", "jpg", "jpeg"], key="sig_upload")
        if up:
            img = Image.open(up)
            img = ensure_transparent_background(img)
            st.image(img, caption="Vista previa (limpiada)", use_container_width=True)
            if st.button("Guardar firma (imagen subida)"):
                save_signature_png(img, SIG_PATH)
                st.success("Firma guardada.")

    if os.path.exists(SIG_PATH):
        st.markdown("**Firma actual guardada:**")
        st.image(SIG_PATH, width=360)


def validate_required_fields(fields: List[Dict[str, Any]], caso: Dict[str, Any]) -> List[str]:
    missing = []
    for f in fields:
        if not f.get("required", False):
            continue
        key = f.get("key")
        if not key:
            continue
        v = caso.get(key, None)
        if f.get("type") == "date":
            if not isinstance(v, date):
                missing.append(f.get("label", key))
        else:
            if v is None or str(v).strip() == "":
                missing.append(f.get("label", key))
    return missing


def render_dynamic_case_form(tpl: Dict[str, Any], perfil: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Renders a dynamic form from tpl["ui"]["fields"].
    Returns (caso dict, fields list used).
    Safe: if ui missing -> returns empty and [].
    """
    ui = tpl.get("ui") or {}
    fields = ui.get("fields") or []
    if not isinstance(fields, list) or len(fields) == 0:
        return {}, []

    # Prefill defaults: Tratante 1 = médico que llena el formulario (si el template pide esos campos)
    prefills = {
        "trt1_nombre": (f"Dr. {perfil.get('nombre','')}".strip() if perfil.get("nombre") else ""),
        "trt1_prof": perfil.get("especialidad", ""),
        "trt1_rut": perfil.get("rut", ""),
        "trt1_tel": perfil.get("telefono", ""),
    }

    caso: Dict[str, Any] = {}
    st.markdown("### Datos del caso")

    # Optional: group by "section"
    sections: Dict[str, List[Dict[str, Any]]] = {}
    for f in fields:
        sec = f.get("section", "Datos")
        sections.setdefault(sec, []).append(f)

    for sec_name, sec_fields in sections.items():
        with st.expander(sec_name, expanded=True):
            for f in sec_fields:
                key = f.get("key")
                if not key:
                    continue
                label = f.get("label", key)
                ftype = (f.get("type") or "text").lower()
                help_txt = f.get("help", None)
                default = f.get("default", None)

                # apply prefills if default not set
                if default is None and key in prefills and prefills[key]:
                    default = prefills[key]

                if ftype == "text":
                    caso[key] = st.text_input(label, value=(default or ""), help=help_txt, key=f"case_{key}")
                elif ftype == "multiline":
                    caso[key] = st.text_area(label, value=(default or ""), height=int(f.get("height_px", 110)), help=help_txt, key=f"case_{key}")
                elif ftype == "date":
                    # Default: si no se define en el JSON, usamos hoy (útil para "fecha del informe").
                    # Importante: NO ponemos límites min/max para permitir cambiar hacia atrás.
                    dflt = parse_date_default(default)
                    caso[key] = st.date_input(label, value=dflt, help=help_txt, key=f"case_{key}")
                elif ftype == "number":
                    minv = f.get("min", None)
                    maxv = f.get("max", None)
                    step = f.get("step", 1)
                    if default is None:
                        default_num = float(minv) if minv is not None else 0.0
                    else:
                        try:
                            default_num = float(default)
                        except Exception:
                            default_num = 0.0
                    caso[key] = st.number_input(
                        label,
                        value=default_num,
                        min_value=float(minv) if minv is not None else None,
                        max_value=float(maxv) if maxv is not None else None,
                        step=float(step),
                        help=help_txt,
                        key=f"case_{key}",
                    )
                elif ftype in ("select", "radio"):
                    options = f.get("options", [])
                    if not isinstance(options, list):
                        options = []
                    idx = options.index(default) if (default in options) else (0 if options else 0)
                    if ftype == "select":
                        caso[key] = st.selectbox(label, options=options, index=idx, help=help_txt, key=f"case_{key}")
                    else:
                        caso[key] = st.radio(label, options=options, index=idx, help=help_txt, key=f"case_{key}", horizontal=bool(f.get("horizontal", False)))
                elif ftype == "checkbox":
                    dflt = default if isinstance(default, bool) else False
                    caso[key] = st.checkbox(label, value=dflt, help=help_txt, key=f"case_{key}")
                else:
                    caso[key] = st.text_input(label, value=(default or ""), help=help_txt, key=f"case_{key}")

    return caso, fields


def render_legacy_case_form() -> Dict[str, Any]:
    """
    Your old fixed UI (so existing templates keep working).
    """
    col1, col2 = st.columns(2)
    with col1:
        paciente_nombre = st.text_input("Nombre paciente", value="")
        paciente_rut = st.text_input("RUT paciente", value="")
        fecha_atencion = st.date_input("Fecha atención", value=date.today())
    with col2:
        fecha_diagnostico = st.date_input("Fecha diagnóstico", value=date.today())
        diagnostico = st.text_input("Diagnóstico (1 línea)", value="")
        tratamiento = st.text_input("Tratamiento (1 línea)", value="")

    return {
        "paciente_nombre": paciente_nombre,
        "paciente_rut": paciente_rut,
        "fecha_atencion": fecha_atencion,
        "fecha_diagnostico": fecha_diagnostico,
        "diagnostico": diagnostico,
        "tratamiento": tratamiento,
    }


# ============================
# App
# ============================
st.set_page_config(page_title="MVP Formularios (LR + UI Dinámica)", layout="wide")
st.title("MVP Formularios ")

if not HAS_FITZ:
    st.error(
        "Falta dependencia: PyMuPDF.\n\n"
        "Instala con:\n"
        "pip install pymupdf\n\n"
        "Luego reinicia: python -m streamlit run app.py"
    )
    st.stop()

perfil = load_profile()
installed = list_installed_templates()

selected_template_id = st.sidebar.selectbox(
    "Template activo",
    options=installed if installed else ["(no hay templates)"],
    index=0
)

tabs = st.tabs([
    "Cliente: Perfil",
    "Cliente: Caso / Generar",
    "Admin: Builder (Marcadores L/R)"
])

# ----------------------------
# Cliente: Perfil
# ----------------------------
with tabs[0]:
    st.subheader("Perfil del médico")
    colA, colB = st.columns(2)
    with colA:
        perfil["nombre"] = st.text_input("Nombre (sin 'Dr.')", value=perfil.get("nombre", ""))
        perfil["rut"] = st.text_input("RUT", value=perfil.get("rut", ""))
    with colB:
        perfil["especialidad"] = st.text_input("Especialidad", value=perfil.get("especialidad", ""))
        perfil["icm"] = st.text_input("ICM (opcional)", value=perfil.get("icm", ""))
        perfil["institucion"] = st.text_input("Institución / Centro", value=perfil.get("institucion", ""))
        perfil["correo"] = st.text_input("Correo electrónico", value=perfil.get("correo", ""))
        perfil["telefono"] = st.text_input("Teléfono", value=perfil.get("telefono", ""))

    perfil["stamp_font_size"] = st.slider("Tamaño fuente timbre", 7, 20, int(perfil.get("stamp_font_size", 9)))

    if st.button("Guardar perfil"):
        save_profile(perfil)
        st.success("Perfil guardado.")

    profile_signature_ui()

# ----------------------------
# Cliente: Caso / Generar
# ----------------------------
with tabs[1]:
    if not installed:
        st.info("No hay templates instalados todavía. Ve a **Admin** y crea el primero.")
    else:
        tpl = load_template(selected_template_id)
        st.subheader(f"Caso + Generación ({tpl.get('name', selected_template_id)})")

        base_pdf_path = template_pdf_path(selected_template_id)
        with open(base_pdf_path, "rb") as f:
            base_pdf_bytes = f.read()

        # Dynamic fields (if present)
        caso_dynamic, ui_fields = render_dynamic_case_form(tpl, perfil)

        if ui_fields:
            caso = caso_dynamic
            st.caption("Esta pantalla se generó dinámicamente desde el template (ui.fields).")
        else:
            st.caption("Este template no tiene ui.fields → usando formulario clásico (compatibilidad).")
            caso = render_legacy_case_form()
        def can_generate() -> Optional[str]:
            if not os.path.exists(SIG_PATH):
                return "Falta firma guardada (Perfil)."

            # Básicos siempre requeridos
            if not perfil.get("nombre") or not perfil.get("rut") or not perfil.get("especialidad"):
                return "Completa el perfil (nombre, RUT, especialidad)."

            # Extras solo si el template los pide (bindings con source perfil.*)
            bindings = (tpl.get("bindings") or {})
            needed = set()
            for b in bindings.values():
                src = str(b.get("source", "")).strip()
                if src == "perfil.institucion":
                    needed.add("institucion")
                elif src == "perfil.correo":
                    needed.add("correo")
                elif src == "perfil.telefono":
                    needed.add("telefono")

            missing_extras = [k for k in sorted(needed) if not str(perfil.get(k, "")).strip()]
            if missing_extras:
                label_map = {"institucion": "institución/centro", "correo": "correo", "telefono": "teléfono"}
                return "Completa el perfil (" + ", ".join(label_map.get(k, k) for k in missing_extras) + ")."

            return None
        # Validate required fields if dynamic
        if ui_fields:
            missing_labels = validate_required_fields(ui_fields, caso)
        else:
            missing_labels = []

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Generar vista previa"):
                err = can_generate()
                if err:
                    st.error(err)
                else:
                    # For preview: if dynamic, create demo values for empties (without changing UI state)
                    caso_preview = dict(caso)
                    # basic fallbacks
                    if "paciente_nombre" in caso_preview and not str(caso_preview.get("paciente_nombre", "")).strip():
                        caso_preview["paciente_nombre"] = "PACIENTE PRUEBA APELLIDO LARGO"
                    if "paciente_rut" in caso_preview and not str(caso_preview.get("paciente_rut", "")).strip():
                        caso_preview["paciente_rut"] = "12.345.678-9"
                    if "diagnostico" in caso_preview and not str(caso_preview.get("diagnostico", "")).strip():
                        caso_preview["diagnostico"] = "DIAGNÓSTICO DE PRUEBA MUY LARGO PARA PROBAR AUTOSHRINK"
                    if "tratamiento" in caso_preview and not str(caso_preview.get("tratamiento", "")).strip():
                        caso_preview["tratamiento"] = "TRATAMIENTO DE PRUEBA MUY LARGO PARA PROBAR AUTOSHRINK"
                    if "fecha_atencion" in caso_preview and not isinstance(caso_preview.get("fecha_atencion"), date):
                        caso_preview["fecha_atencion"] = date.today()
                    if "fecha_diagnostico" in caso_preview and not isinstance(caso_preview.get("fecha_diagnostico"), date):
                        caso_preview["fecha_diagnostico"] = date.today()

                    overlay_reader = make_overlay(
                        PdfReader(io.BytesIO(base_pdf_bytes)),
                        tpl,
                        caso_preview,
                        perfil,
                        SIG_PATH
                    )
                    out_bytes = merge_overlay(base_pdf_bytes, overlay_reader)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"preview_{selected_template_id}_{ts}.pdf"
                    out_path = os.path.join(OUT_DIR, fname)
                    with open(out_path, "wb") as f:
                        f.write(out_bytes)

                    st.download_button("Descargar vista previa", data=out_bytes, file_name=fname, mime="application/pdf")
                    st.success("Vista previa generada.")

        with colB:
            if st.button("Generar PDF final"):
                err = can_generate()
                if err:
                    st.error(err)
                else:
                    if ui_fields and missing_labels:
                        st.error("Faltan campos requeridos: " + ", ".join(missing_labels))
                    else:
                        # legacy minimal validation (soft)
                        if not ui_fields:
                            legacy_missing = [k for k in ["paciente_nombre", "paciente_rut", "diagnostico"] if not str(caso.get(k, "")).strip()]
                            if legacy_missing:
                                st.error(f"Faltan campos: {', '.join(legacy_missing)}")
                                st.stop()

                        overlay_reader = make_overlay(
                            PdfReader(io.BytesIO(base_pdf_bytes)),
                            tpl,
                            caso,
                            perfil,
                            SIG_PATH
                        )
                        out_bytes = merge_overlay(base_pdf_bytes, overlay_reader)

                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = f"final_{selected_template_id}_{ts}.pdf"
                        out_path = os.path.join(OUT_DIR, fname)
                        with open(out_path, "wb") as f:
                            f.write(out_bytes)

                        st.download_button("Descargar PDF final", data=out_bytes, file_name=fname, mime="application/pdf")
                        st.success("PDF final generado.")

        st.markdown("---")
        st.markdown("### Historial (outputs)")
        outs = sorted([f for f in os.listdir(OUT_DIR) if f.lower().endswith(".pdf")], reverse=True)
        for fname in outs[:10]:
            p = os.path.join(OUT_DIR, fname)
            with open(p, "rb") as f:
                st.download_button(f"Descargar {fname}", f, file_name=fname, mime="application/pdf", key=f"dl_{fname}")

# ----------------------------
# Admin: Builder (Marcadores L/R)
# ----------------------------
with tabs[2]:
    st.subheader("Admin: Builder por marcadores L/R (sin clave)")

    st.markdown("## Marcadores recomendados (cortos)")
    st.code("\n".join(DEFAULT_MARKERS))
    st.caption(
        "Pon cada par L/R en la misma línea del campo.\n"
        "Ejemplo: en Diagnóstico pones <<dg_l>> al inicio y <<dg_r>> al final.\n"
        "La app calcula el ancho exacto como distancia entre ambos."
    )

    st.markdown("## Crear template")
    up_marked = st.file_uploader("1) PDF marcado (obligatorio)", type=["pdf"], key="marked_pdf")
    up_base = st.file_uploader("2) PDF base limpio (opcional, recomendado)", type=["pdf"], key="base_pdf")

    c1, c2, c3 = st.columns(3)
    with c1:
        new_tid = st.text_input("template_id (ej: bice_v1)", value="")
    with c2:
        new_name = st.text_input("Nombre visible", value="")
    with c3:
        new_vendor = st.text_input("Aseguradora/Vendor", value="")

    markers_text = st.text_area(
        "Marcadores a buscar (uno por línea).",
        value="\n".join(DEFAULT_MARKERS),
        height=180
    )
    markers = [m.strip() for m in markers_text.splitlines() if m.strip()]

    if st.button("Detectar marcadores y crear template"):
        if not up_marked:
            st.error("Debes subir el PDF marcado.")
        else:
            tid = sanitize_template_id(new_tid)
            if not tid:
                st.error("template_id inválido.")
            elif os.path.exists(template_json_path(tid)) or os.path.exists(template_pdf_path(tid)):
                st.error("Ya existe un template con ese id.")
            else:
                pdf_marked_bytes = up_marked.getvalue()
                pdf_base_bytes = up_base.getvalue() if up_base else None

                hits = find_markers_in_pdf(pdf_marked_bytes, markers)
                found = {m: v for m, v in hits.items() if v}
                missing = [m for m in markers if not hits.get(m)]

                if not found:
                    st.error("No se encontró ningún marcador. Revisa que el texto sea exactamente igual y seleccionable.")
                else:
                    build_template(
                        template_id=tid,
                        pdf_marked_bytes=pdf_marked_bytes,
                        pdf_base_bytes=pdf_base_bytes,
                        name=new_name,
                        vendor=new_vendor,
                        markers=markers
                    )
                    st.success(f"Template creado: {tid}. Marcadores detectados: {len(found)}.")
                    if up_base:
                        st.success("PDF base limpio guardado → NO se tapará el formulario.")
                    else:
                        st.warning("No subiste PDF base limpio → se usará el marcado como base y se taparán los marcadores con blanco.")
                    if missing:
                        st.warning("Marcadores NO encontrados:\n" + "\n".join(missing))
                    st.rerun()

    st.markdown("---")
    st.markdown("## Ajustes del template (altura campos / sello-firma)")
    installed_now = list_installed_templates()
    if not installed_now:
        st.info("Aún no hay templates.")
    else:
        tsel = st.selectbox("Template a editar", options=installed_now, key="edit_tpl")
        tpl = load_template(tsel)

        bindings = tpl.get("bindings", DEFAULT_BINDINGS)
        style = tpl.get("style", {})
        st.write("Base limpio:", "Sí" if not tpl.get("needs_cover", False) else "No (tapa con blanco)")

        st.markdown("### Altura (pts) para campos 1 línea")
        h = st.slider("Altura por defecto (pts)", 8.0, 22.0, 12.0, 0.5)

        if st.button("Aplicar altura a todos los campos 1 línea"):
            for k, b in bindings.items():
                if b.get("type") == "lr_singleline":
                    b["height"] = float(h)
            tpl["bindings"] = bindings
            tpl["version"] = int(tpl.get("version", 1)) + 1
            save_template(tsel, tpl)
            st.success("Altura aplicada.")
            st.rerun()

        st.markdown("### Sello/Firma (traslape)")
        sf = bindings.get("sello_firma", DEFAULT_BINDINGS["sello_firma"])
        colS1, colS2, colS3 = st.columns(3)
        with colS1:
            sf["box_w"] = st.slider("SF box_w", 100, 420, int(sf.get("box_w", 200)))
            sf["box_h"] = st.slider("SF box_h", 40, 240, int(sf.get("box_h", 80)))
        with colS2:
            style["stamp_y_ratio"] = st.slider("stamp_y_ratio", 0.05, 0.55, float(style.get("stamp_y_ratio", 0.22)))
            style["sig_h_ratio"] = st.slider("sig_h_ratio", 0.50, 1.00, float(style.get("sig_h_ratio", 0.85)))
        with colS3:
            style["stamp_gap"] = st.slider("stamp_gap", 0.0, 5.0, float(style.get("stamp_gap", 1.0)))
            style["cover_padding"] = st.slider("cover_padding (si no hay base limpio)", 0.0, 6.0, float(style.get("cover_padding", 1.5)))

        bindings["sello_firma"] = sf
        tpl["bindings"] = bindings
        tpl["style"] = style

        if st.button("Guardar cambios"):
            tpl["version"] = int(tpl.get("version", 1)) + 1
            save_template(tsel, tpl)
            st.success("Guardado.")

    st.markdown("---")
    st.markdown("## Gestión: Importar / Exportar templates")
    colA, colB = st.columns(2)
    with colA:
        upzip = st.file_uploader("Importar ZIP de template(s)", type=["zip"], key="import_zip")
        if upzip and st.button("Importar ZIP"):
            ids = import_template_zip(upzip.getvalue())
            if ids:
                st.success(f"Instalados: {', '.join(ids)}")
                st.rerun()
            else:
                st.error("ZIP no contiene templates válidos.")
    with colB:
        if installed_now:
            tselx = st.selectbox("Exportar template", options=installed_now, key="export_sel")
            if st.button("Generar ZIP"):
                zbytes = export_template_zip_bytes(tselx)
                st.download_button("Descargar ZIP", data=zbytes, file_name=f"{tselx}.zip", mime="application/zip")
