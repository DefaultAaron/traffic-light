"""S2TLD annotation helper \u2014 view and edit Pascal VOC XML annotations.

Layout: Image (left) | Annotation panel (right)

Controls:
    Image focused:  \u2191/\u2193 = prev/next image
    Ctrl+\u2191/\u2193:      prev/next image (works everywhere)
    Scroll/+/-:     Zoom in/out
    Ctrl+0:         Reset zoom to fit
    A:              Toggle all annotation overlays
    Click+drag:     Draw bounding box, then select class to add
    Right/Middle-drag: Pan image (when zoomed)
    Ctrl+g:         Jump to Nth image
    Ctrl+q:         Quit

Auto-saves after 500ms of inactivity.
Unsaved changes are always flushed before navigation and on quit.
"""

import sys
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "raw" / "S2TLD" / "JPEGImages"
ANNOT_DIR = ROOT / "data" / "raw" / "S2TLD" / "Annotations-fix"

# 7 classes \u2014 R/Y/G round + R/G left/right (RGB tuples for PIL)
CLASS_COLORS = {
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "redLeft": (200, 0, 0),
    "greenLeft": (0, 200, 0),
    "redRight": (255, 0, 128),
    "greenRight": (0, 255, 128),
    "off": (128, 128, 128),
    "wait_on": (200, 200, 200),
}

# Classes available for annotation
ANNOTATABLE_CLASSES = [
    "red", "yellow", "green",
    "redLeft", "greenLeft",
    "redRight", "greenRight",
    "off", "wait_on",
]


def parse_annotation(xml_text: str) -> list[dict]:
    """Parse Pascal VOC XML string.

    Returns list of dicts with keys: name, xmin, ymin, xmax, ymax,
    plus pose, truncated, difficult (preserved for lossless roundtrip).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            objects.append({
                "name": name,
                "xmin": int(bbox.findtext("xmin", "0")),
                "ymin": int(bbox.findtext("ymin", "0")),
                "xmax": int(bbox.findtext("xmax", "0")),
                "ymax": int(bbox.findtext("ymax", "0")),
                "pose": obj.findtext("pose", "Unspecified"),
                "truncated": obj.findtext("truncated", "0"),
                "difficult": obj.findtext("difficult", "0"),
            })
        except ValueError:
            continue
    return objects


def draw_annotations(
    img: Image.Image, objects: list[dict], visible: list[bool]
) -> Image.Image:
    """Draw bounding boxes and labels for objects where visible[i] is True."""
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except (OSError, AttributeError):
        label_font = ImageFont.load_default()

    for i, obj in enumerate(objects):
        if not visible[i]:
            continue
        color = CLASS_COLORS.get(obj["name"], (255, 255, 255))
        x1, y1, x2, y2 = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"[{i}] {obj['name']}"
        bbox = draw.textbbox((x1, y1), label, font=label_font)
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0), font=label_font)
    return canvas


def objects_to_xml(xml_text: str, objects: list[dict]) -> str:
    """Replace all <object> elements in the XML with the given list.

    Preserves pose/truncated/difficult from each object dict to avoid
    silently overwriting original values.
    """
    try:
        root = ET.fromstring(xml_text)
        for obj in root.findall("object"):
            root.remove(obj)
    except ET.ParseError:
        # XML is corrupt or empty — build a minimal skeleton so edits are not lost
        root = ET.Element("annotation")

    for o in objects:
        obj_el = ET.SubElement(root, "object")
        ET.SubElement(obj_el, "name").text = o["name"]
        ET.SubElement(obj_el, "pose").text = o.get("pose", "Unspecified")
        ET.SubElement(obj_el, "truncated").text = o.get("truncated", "0")
        ET.SubElement(obj_el, "difficult").text = o.get("difficult", "0")
        bndbox = ET.SubElement(obj_el, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(o["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(o["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(o["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(o["ymax"])

    ET.indent(root, space="\t")
    return ET.tostring(root, encoding="unicode") + "\n"


def _color_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


class AnnotationApp:
    def __init__(self, root: tk.Tk, pairs: list[tuple[Path, Path]]):
        self.root = root
        self.pairs = pairs
        self.idx = 0
        self.save_job = None
        self.dirty = False
        self._current_img = None
        self._current_objects: list[dict] = []
        self._current_xml_text = ""  # full XML text (source of truth for disk)

        # Per-annotation visibility (no master flag \u2014 avoids the bug where
        # individual toggles are silently ignored after "Hide All")
        self._visible: list[bool] = []

        # Zoom state
        self._zoom_level = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start = None

        # Drawing state for new bounding box
        self._drawing = False
        self._draw_start = None
        self._draw_rect_id = None

        self.root.title("S2TLD Annotation Helper")
        self.root.configure(bg="#2b2b2b")

        # --- Top bar ---
        topbar = tk.Frame(root, bg="#333333", pady=4)
        topbar.pack(fill=tk.X)

        btn_prev = tk.Button(topbar, text="\u25c0 Prev", command=self.prev_image,
                             bg="#555", fg="white", relief=tk.FLAT, padx=8)
        btn_prev.pack(side=tk.LEFT, padx=6)

        self.nav_label = tk.Label(topbar, text="", bg="#333333", fg="white",
                                  font=("Menlo", 13))
        self.nav_label.pack(side=tk.LEFT, padx=10)

        btn_next = tk.Button(topbar, text="Next \u25b6", command=self.next_image,
                             bg="#555", fg="white", relief=tk.FLAT, padx=8)
        btn_next.pack(side=tk.LEFT, padx=6)

        # Jump-to box: type image number and press Enter
        tk.Label(topbar, text="Go:", bg="#333333", fg="#aaaaaa",
                 font=("Menlo", 11)).pack(side=tk.LEFT, padx=(10, 2))
        self.jump_entry = tk.Entry(topbar, width=6, font=("Menlo", 11),
                                   bg="#2a2a2a", fg="white", insertbackground="white",
                                   highlightbackground="#555555", highlightcolor="#66bb6a")
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Return>", lambda e: self._jump_to_image())

        self.zoom_label = tk.Label(topbar, text="100%", bg="#333333", fg="#aaaaaa",
                                   font=("Menlo", 11))
        self.zoom_label.pack(side=tk.LEFT, padx=10)

        self.annot_toggle_btn = tk.Button(
            topbar, text="[A] Hide All", command=self._toggle_all_annotations,
            bg="#555", fg="#66bb6a", relief=tk.FLAT, padx=8, font=("Menlo", 10))
        self.annot_toggle_btn.pack(side=tk.LEFT, padx=6)

        self.status_label = tk.Label(topbar, text="", bg="#333333", fg="#888888",
                                     font=("Menlo", 11))
        self.status_label.pack(side=tk.RIGHT, padx=10)

        self.focus_hint = tk.Label(topbar, text="", bg="#333333", fg="#555555",
                                   font=("Menlo", 10))
        self.focus_hint.pack(side=tk.RIGHT, padx=10)

        # --- Main content: image left, annotation panel right ---
        content = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#2b2b2b",
                                 sashwidth=4, sashrelief=tk.RAISED)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: image canvas
        self.img_canvas = tk.Canvas(content, bg="#1e1e1e", highlightthickness=0)
        content.add(self.img_canvas, stretch="always")
        self.img_canvas.bind("<Configure>", self._on_canvas_resize)
        self.img_canvas.bind("<FocusIn>", lambda e: self._update_focus_hint())
        self.img_canvas.bind("<Up>", lambda e: self.prev_image())
        self.img_canvas.bind("<Down>", lambda e: self.next_image())
        self.img_canvas.bind("<Button-1>", self._on_canvas_click)
        self.img_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.img_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        # Zoom: scroll wheel
        self.img_canvas.bind("<MouseWheel>", self._on_scroll_zoom)
        self.img_canvas.bind("<Button-4>", lambda e: self._zoom(1.1, e.x, e.y))
        self.img_canvas.bind("<Button-5>", lambda e: self._zoom(0.9, e.x, e.y))

        # Zoom: keyboard +/-
        self.img_canvas.bind("<plus>", lambda e: self._zoom(1.2))
        self.img_canvas.bind("<minus>", lambda e: self._zoom(0.8))
        self.img_canvas.bind("<equal>", lambda e: self._zoom(1.2))

        # Pan: middle-click or right-click drag (right-click = trackpad two-finger click)
        self.img_canvas.bind("<Button-2>", self._on_pan_start)
        self.img_canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.img_canvas.bind("<Button-3>", self._on_pan_start)
        self.img_canvas.bind("<B3-Motion>", self._on_pan_drag)

        # Toggle annotations: A key
        self.img_canvas.bind("<a>", lambda e: self._toggle_all_annotations())

        # Right: annotation panel
        self._build_annotation_panel(content)

        # Global key bindings
        self.root.bind("<Control-Up>", lambda e: self.prev_image())
        self.root.bind("<Control-Down>", lambda e: self.next_image())
        self.root.bind("<Control-q>", lambda e: self._quit())
        self.root.bind("<Control-Key-0>", lambda e: self._reset_zoom())
        self.root.bind("<Control-g>", lambda e: self._focus_jump_entry())

        # Ensure _flush_save runs even if user clicks the window X button
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.root.after(100, lambda: self.img_canvas.focus_set())
        self.load_current()

    # ------------------------------------------------------------------
    # Annotation panel (right side)
    # ------------------------------------------------------------------

    def _build_annotation_panel(self, parent: tk.PanedWindow):
        panel = tk.Frame(parent, bg="#2b2b2b")
        parent.add(panel, width=420, stretch="never")

        # Header
        hdr = tk.Frame(panel, bg="#2b2b2b")
        hdr.pack(fill=tk.X, padx=4, pady=(6, 2))

        tk.Label(hdr, text="Annotations", bg="#2b2b2b", fg="#aaaaaa",
                 font=("Menlo", 12, "bold"), anchor=tk.W).pack(side=tk.LEFT)

        self.count_label = tk.Label(hdr, text="(0)", bg="#2b2b2b", fg="#666666",
                                    font=("Menlo", 11))
        self.count_label.pack(side=tk.LEFT, padx=6)

        # Scrollable annotation list
        list_container = tk.Frame(panel, bg="#2b2b2b")
        list_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self._list_canvas = tk.Canvas(list_container, bg="#1e1e1e",
                                      highlightthickness=0, borderwidth=0)
        scrollbar = tk.Scrollbar(list_container, command=self._list_canvas.yview)
        self._list_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._list_inner = tk.Frame(self._list_canvas, bg="#1e1e1e")
        self._list_canvas_window = self._list_canvas.create_window(
            (0, 0), window=self._list_inner, anchor=tk.NW)

        self._list_inner.bind("<Configure>",
                              lambda e: self._list_canvas.configure(
                                  scrollregion=self._list_canvas.bbox("all")))
        self._list_canvas.bind("<Configure>", self._on_list_canvas_resize)

        # Enable scroll wheel on annotation list
        self._list_canvas.bind("<MouseWheel>", self._on_list_scroll)
        self._list_inner.bind("<MouseWheel>", self._on_list_scroll)

        # Row widgets cache
        self._row_widgets: list[dict] = []

    def _on_list_canvas_resize(self, event):
        self._list_canvas.itemconfigure(self._list_canvas_window, width=event.width)

    def _on_list_scroll(self, event):
        self._list_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _rebuild_annotation_list(self):
        """Rebuild the annotation list panel from self._current_objects."""
        for w in self._row_widgets:
            w["frame"].destroy()
        self._row_widgets.clear()

        self.count_label.config(text=f"({len(self._current_objects)})")
        self._update_toggle_btn()

        for i, obj in enumerate(self._current_objects):
            self._create_row(i, obj)

    def _create_row(self, idx: int, obj: dict):
        color_rgb = CLASS_COLORS.get(obj["name"], (255, 255, 255))
        color_hex = _color_hex(color_rgb)
        bg_hex = _color_hex(tuple(max(0, c // 8 + 20) for c in color_rgb))

        row = tk.Frame(self._list_inner, bg=bg_hex, padx=4, pady=3,
                       highlightbackground="#444444", highlightthickness=1)
        row.pack(fill=tk.X, padx=2, pady=1)
        row.bind("<MouseWheel>", self._on_list_scroll)

        # Top line: checkbox + index + class label + delete
        top = tk.Frame(row, bg=bg_hex)
        top.pack(fill=tk.X)
        top.bind("<MouseWheel>", self._on_list_scroll)

        # Visibility checkbox
        vis_var = tk.BooleanVar(value=self._visible[idx] if idx < len(self._visible) else True)
        chk = tk.Checkbutton(top, variable=vis_var, bg=bg_hex, activebackground=bg_hex,
                             command=lambda i=idx, v=vis_var: self._on_toggle_visible(i, v))
        chk.pack(side=tk.LEFT)

        # Index label
        tk.Label(top, text=f"[{idx}]", bg=bg_hex, fg="#888888",
                 font=("Menlo", 11)).pack(side=tk.LEFT, padx=(0, 4))

        # Class label (clickable to change)
        cls_btn = tk.Button(top, text=obj["name"], bg=bg_hex, fg=color_hex,
                            font=("Menlo", 12, "bold"), relief=tk.FLAT,
                            activebackground=bg_hex, activeforeground="white",
                            cursor="hand2",
                            command=lambda i=idx: self._change_class(i))
        cls_btn.pack(side=tk.LEFT, padx=2)

        # Delete button
        del_btn = tk.Button(top, text="\u2715", bg=bg_hex, fg="#ef5350",
                            font=("Menlo", 11), relief=tk.FLAT,
                            activebackground="#ef5350", activeforeground="white",
                            cursor="hand2", padx=4,
                            command=lambda i=idx: self._delete_annotation(i))
        del_btn.pack(side=tk.RIGHT, padx=2)

        # Bottom line: bbox coords (clickable to edit)
        coords = f"({obj['xmin']}, {obj['ymin']}) \u2192 ({obj['xmax']}, {obj['ymax']})  " \
                 f"[{obj['xmax'] - obj['xmin']}\u00d7{obj['ymax'] - obj['ymin']}]"
        coord_lbl = tk.Label(row, text=coords, bg=bg_hex, fg="#999999",
                             font=("Menlo", 10), anchor=tk.W, cursor="hand2")
        coord_lbl.pack(fill=tk.X, padx=(26, 0))
        coord_lbl.bind("<MouseWheel>", self._on_list_scroll)
        coord_lbl.bind("<Button-1>", lambda e, i=idx: self._start_edit_box(i))

        self._row_widgets.append({
            "frame": row,
            "vis_var": vis_var,
            "cls_btn": cls_btn,
            "coord_lbl": coord_lbl,
        })

    def _on_toggle_visible(self, idx: int, var: tk.BooleanVar):
        if idx < len(self._visible):
            self._visible[idx] = var.get()
        self._update_toggle_btn()
        self._render_image()

    def _toggle_all_annotations(self):
        """Toggle all: if any visible \u2192 hide all, else \u2192 show all."""
        any_visible = any(self._visible)
        new_val = not any_visible
        for i in range(len(self._visible)):
            self._visible[i] = new_val
        for w in self._row_widgets:
            w["vis_var"].set(new_val)
        self._update_toggle_btn()
        self._render_image()

    def _update_toggle_btn(self):
        """Update the toggle-all button label based on current visibility."""
        if any(self._visible):
            self.annot_toggle_btn.config(text="[A] Hide All", fg="#66bb6a")
        else:
            self.annot_toggle_btn.config(text="[A] Show All", fg="#ef5350")

    def _change_class(self, idx: int):
        if idx >= len(self._current_objects):
            return
        w = self._row_widgets[idx]["cls_btn"]
        x = w.winfo_rootx()
        y = w.winfo_rooty() + w.winfo_height()

        menu = tk.Menu(self.root, tearoff=0, bg="#333333", fg="white",
                       activebackground="#555555", activeforeground="white",
                       font=("Menlo", 12))
        for cls_name in ANNOTATABLE_CLASSES:
            menu.add_command(
                label=cls_name,
                command=lambda c=cls_name, i=idx: self._set_class(i, c))
        menu.post(x, y)

    def _set_class(self, idx: int, cls_name: str):
        if idx >= len(self._current_objects):
            return
        old = self._current_objects[idx]["name"]
        if old == cls_name:
            return
        self._current_objects[idx]["name"] = cls_name
        self._sync_and_mark_dirty()
        self._rebuild_annotation_list()
        self._render_image()
        self.status_label.config(text=f"[{idx}] {old} \u2192 {cls_name}", fg="#66bb6a")

    def _start_edit_box(self, idx: int):
        """Replace coord label with inline entry fields for editing bbox."""
        if idx >= len(self._current_objects) or idx >= len(self._row_widgets):
            return
        obj = self._current_objects[idx]
        w = self._row_widgets[idx]
        coord_lbl = w.get("coord_lbl")
        if coord_lbl is None:
            return
        coord_lbl.pack_forget()

        bg_hex = coord_lbl.cget("bg")
        edit_frame = tk.Frame(w["frame"], bg=bg_hex)
        edit_frame.pack(fill=tk.X, padx=(26, 0))

        entries = {}
        for key in ("xmin", "ymin", "xmax", "ymax"):
            tk.Label(edit_frame, text=key, bg=bg_hex, fg="#888888",
                     font=("Menlo", 9)).pack(side=tk.LEFT)
            e = tk.Entry(edit_frame, width=5, font=("Menlo", 10),
                         bg="#2a2a2a", fg="white", insertbackground="white",
                         highlightbackground="#555555", highlightcolor="#66bb6a")
            e.insert(0, str(obj[key]))
            e.pack(side=tk.LEFT, padx=(1, 4))
            e.bind("<Return>", lambda ev, i=idx, f=edit_frame: self._confirm_edit_box(i, f))
            e.bind("<Escape>", lambda ev: self._rebuild_annotation_list())
            entries[key] = e

        tk.Button(edit_frame, text="\u2713", bg="#388e3c", fg="white",
                  font=("Menlo", 10), relief=tk.FLAT, cursor="hand2", padx=3,
                  command=lambda: self._confirm_edit_box(idx, edit_frame)).pack(side=tk.LEFT, padx=2)
        tk.Button(edit_frame, text="\u2717", bg="#c62828", fg="white",
                  font=("Menlo", 10), relief=tk.FLAT, cursor="hand2", padx=3,
                  command=lambda: self._rebuild_annotation_list()).pack(side=tk.LEFT)

        w["_edit_frame"] = edit_frame
        w["_edit_entries"] = entries
        entries["xmin"].focus_set()
        entries["xmin"].select_range(0, tk.END)

    def _confirm_edit_box(self, idx: int, edit_frame: tk.Frame):
        """Validate and apply edited bbox coordinates."""
        if idx >= len(self._current_objects) or idx >= len(self._row_widgets):
            return
        entries = self._row_widgets[idx].get("_edit_entries")
        if entries is None:
            return
        try:
            vals = {k: int(e.get()) for k, e in entries.items()}
        except ValueError:
            edit_frame.bell()
            return
        if vals["xmin"] >= vals["xmax"] or vals["ymin"] >= vals["ymax"]:
            edit_frame.bell()
            return
        obj = self._current_objects[idx]
        if all(obj[k] == vals[k] for k in vals):
            self._rebuild_annotation_list()
            return
        for k, v in vals.items():
            obj[k] = v
        self._sync_and_mark_dirty()
        self._rebuild_annotation_list()
        self._render_image()
        self.status_label.config(
            text=f"[{idx}] box \u2192 ({vals['xmin']},{vals['ymin']})-({vals['xmax']},{vals['ymax']})",
            fg="#66bb6a")

    def _delete_annotation(self, idx: int):
        if idx >= len(self._current_objects):
            return
        removed = self._current_objects.pop(idx)
        self._visible.pop(idx)
        self._sync_and_mark_dirty()
        self._rebuild_annotation_list()
        self._render_image()
        self.status_label.config(
            text=f"Deleted [{idx}] {removed['name']}", fg="#ef5350")

    # ------------------------------------------------------------------
    # Zoom / Pan
    # ------------------------------------------------------------------

    def _on_scroll_zoom(self, event):
        if event.delta > 0:
            self._zoom(1.1, event.x, event.y)
        elif event.delta < 0:
            self._zoom(0.9, event.x, event.y)

    def _zoom(self, factor, cx=None, cy=None):
        old_zoom = self._zoom_level
        self._zoom_level = max(0.1, min(20.0, self._zoom_level * factor))

        if cx is not None and cy is not None and self._current_img is not None:
            cw = self.img_canvas.winfo_width()
            ch = self.img_canvas.winfo_height()
            img_w, img_h = self._current_img.size
            fit_scale = min(cw / img_w, ch / img_h)

            old_scale = fit_scale * old_zoom
            new_scale = fit_scale * self._zoom_level

            img_x = (cx - cw / 2) / old_scale + img_w / 2 - self._pan_x / old_scale
            img_y = (cy - ch / 2) / old_scale + img_h / 2 - self._pan_y / old_scale

            self._pan_x = (img_w / 2 - img_x) * new_scale + (cx - cw / 2)
            self._pan_y = (img_h / 2 - img_y) * new_scale + (cy - ch / 2)

        self._update_zoom_label()
        self._render_image()

    def _reset_zoom(self):
        self._zoom_level = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_zoom_label()
        self._render_image()

    def _update_zoom_label(self):
        pct = int(self._zoom_level * 100)
        self.zoom_label.config(text=f"{pct}%")

    def _on_pan_start(self, event):
        self._drag_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._drag_start is None:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)
        self._pan_x += dx
        self._pan_y += dy
        self._render_image()

    # ------------------------------------------------------------------
    # Drawing new bounding box
    # ------------------------------------------------------------------

    def _canvas_to_image(self, cx, cy) -> tuple[int, int] | None:
        if self._current_img is None:
            return None
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        img_w, img_h = self._current_img.size
        fit_scale = min(cw / img_w, ch / img_h)
        scale = fit_scale * self._zoom_level

        img_cx = (cw - img_w * scale) / 2 + self._pan_x
        img_cy = (ch - img_h * scale) / 2 + self._pan_y

        ix = int((cx - img_cx) / scale)
        iy = int((cy - img_cy) / scale)
        return ix, iy

    def _clear_pending_rect(self):
        """Remove the selection rectangle left over from a completed drag."""
        if self._draw_rect_id:
            self.img_canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

    def _on_canvas_click(self, event):
        self.img_canvas.focus_set()
        self._clear_pending_rect()
        self._drawing = True
        self._draw_start = (event.x, event.y)

    def _on_canvas_drag(self, event):
        if not self._drawing or self._draw_start is None:
            return
        if self._draw_rect_id:
            self.img_canvas.delete(self._draw_rect_id)
        x0, y0 = self._draw_start
        self._draw_rect_id = self.img_canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#00ffff", width=2, dash=(4, 4))

    def _on_canvas_release(self, event):
        if not self._drawing or self._draw_start is None:
            self._drawing = False
            return

        # Finalize the rectangle as a solid outline (keep visible during class pick)
        if self._draw_rect_id:
            self.img_canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None

        x0, y0 = self._draw_start
        x1, y1 = event.x, event.y
        self._drawing = False
        self._draw_start = None

        if abs(x1 - x0) < 5 and abs(y1 - y0) < 5:
            return

        # Draw a solid selection rectangle that persists until class is picked
        self._draw_rect_id = self.img_canvas.create_rectangle(
            x0, y0, x1, y1, outline="#00ffff", width=2)

        pt0 = self._canvas_to_image(x0, y0)
        pt1 = self._canvas_to_image(x1, y1)
        if pt0 is None or pt1 is None:
            return

        img_w, img_h = self._current_img.size
        ix0 = max(0, min(pt0[0], pt1[0]))
        iy0 = max(0, min(pt0[1], pt1[1]))
        ix1 = min(img_w, max(pt0[0], pt1[0]))
        iy1 = min(img_h, max(pt0[1], pt1[1]))

        if ix1 - ix0 < 2 or iy1 - iy0 < 2:
            return

        self._show_class_picker(event.x_root, event.y_root, ix0, iy0, ix1, iy1)

    def _show_class_picker(self, screen_x: int, screen_y: int,
                           xmin: int, ymin: int, xmax: int, ymax: int):
        menu = tk.Menu(self.root, tearoff=0, bg="#333333", fg="white",
                       activebackground="#555555", activeforeground="white",
                       font=("Menlo", 12))

        for cls_name in ANNOTATABLE_CLASSES:
            menu.add_command(
                label=cls_name,
                command=lambda c=cls_name: self._add_annotation(c, xmin, ymin, xmax, ymax))

        menu.add_separator()
        menu.add_command(label="Cancel", command=self._clear_pending_rect)
        # Clean up selection rect when menu is dismissed by clicking elsewhere
        menu.bind("<Unmap>", lambda e: self._clear_pending_rect())
        menu.post(screen_x, screen_y)

    def _add_annotation(self, cls_name: str, xmin: int, ymin: int, xmax: int, ymax: int):
        new_obj = {
            "name": cls_name,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "pose": "Unspecified", "truncated": "0", "difficult": "0",
        }
        self._current_objects.append(new_obj)
        self._visible.append(True)
        self._sync_and_mark_dirty()
        self._rebuild_annotation_list()
        self._render_image()
        self.status_label.config(
            text=f"Added: {cls_name} ({xmin},{ymin})-({xmax},{ymax})", fg="#66bb6a")

    # ------------------------------------------------------------------
    # XML serialization + dirty tracking
    # ------------------------------------------------------------------

    def _sync_and_mark_dirty(self):
        """Rewrite XML text from current objects and schedule a save."""
        self._current_xml_text = objects_to_xml(self._current_xml_text, self._current_objects)
        self._schedule_save()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _schedule_save(self):
        if self.save_job:
            self.root.after_cancel(self.save_job)
        self.save_job = self.root.after(500, self._auto_save)
        self.dirty = True

    def _auto_save(self):
        self.save_job = None
        self._do_save()

    def _do_save(self):
        """Write current XML to disk. Only writes if dirty."""
        if not self.dirty:
            return
        _, xml_path = self.pairs[self.idx]
        text = self._current_xml_text
        try:
            xml_path.write_text(text, encoding="utf-8")
            self.dirty = False
            self.status_label.config(text="Saved", fg="#66bb6a")
            self.root.after(2000, lambda: self.status_label.config(text="", fg="#888888"))
        except OSError as e:
            self.status_label.config(text=f"Save error: {e}", fg="#ef5350")

    def _flush_save(self):
        """Cancel any pending auto-save timer and write immediately if dirty."""
        if self.save_job:
            self.root.after_cancel(self.save_job)
            self.save_job = None
        self._do_save()

    # ------------------------------------------------------------------
    # Image rendering
    # ------------------------------------------------------------------

    def _on_canvas_resize(self, event=None):
        self._render_image()

    def _render_image(self):
        if self._current_img is None:
            return

        objects = self._current_objects or []
        if objects and self._visible:
            display_img = draw_annotations(self._current_img, objects, self._visible)
        else:
            display_img = self._current_img.copy()

        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        img_w, img_h = display_img.size
        fit_scale = min(cw / img_w, ch / img_h)
        scale = fit_scale * self._zoom_level
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = display_img.resize((new_w, new_h), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(resized)
        self.img_canvas.delete("all")
        x = (cw - new_w) // 2 + int(self._pan_x)
        y = (ch - new_h) // 2 + int(self._pan_y)
        self.img_canvas.create_image(x, y, anchor=tk.NW, image=self._tk_img)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def load_current(self):
        img_path, xml_path = self.pairs[self.idx]

        # Reset zoom/pan
        self._zoom_level = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._update_zoom_label()

        # Nav label
        self.nav_label.config(
            text=f"[{self.idx + 1}/{len(self.pairs)}]  {img_path.name}")

        try:
            self._current_img = Image.open(img_path)
        except OSError:
            self.status_label.config(text=f"Cannot read: {img_path.name}", fg="#ef5350")
            self._current_img = None
            return

        try:
            xml_text = xml_path.read_text(encoding="utf-8")
        except OSError:
            xml_text = ""

        self._current_xml_text = xml_text
        self._current_objects = parse_annotation(xml_text)
        self._visible = [True] * len(self._current_objects)
        self.dirty = False

        self._rebuild_annotation_list()
        self._render_image()
        self.status_label.config(text="", fg="#888888")

    def _update_focus_hint(self):
        focused = self.root.focus_get()
        if focused == self.img_canvas:
            self.focus_hint.config(text="[Image] \u2191\u2193=nav  Scroll=zoom  Drag=box")
        else:
            self.focus_hint.config(text="")

    def _focus_jump_entry(self):
        self.jump_entry.focus_set()
        self.jump_entry.select_range(0, tk.END)

    def _jump_to_image(self):
        """Jump to the Nth image pair (1-based index from the entry field)."""
        text = self.jump_entry.get().strip()
        try:
            n = int(text)
        except ValueError:
            self.status_label.config(text=f"Invalid number: {text}", fg="#ef5350")
            return
        if n < 1 or n > len(self.pairs):
            self.status_label.config(
                text=f"Out of range (1-{len(self.pairs)})", fg="#ef5350")
            return
        if n - 1 == self.idx:
            return
        self._flush_save()
        if self.dirty:
            return
        self.idx = n - 1
        self.load_current()
        self.img_canvas.focus_set()

    def prev_image(self):
        self._flush_save()
        if self.dirty:
            return  # save failed — stay on current image to avoid data loss
        self.idx = (self.idx - 1) % len(self.pairs)
        self.load_current()

    def next_image(self):
        self._flush_save()
        if self.dirty:
            return  # save failed — stay on current image to avoid data loss
        self.idx = (self.idx + 1) % len(self.pairs)
        self.load_current()

    def _quit(self):
        self._flush_save()
        self.root.quit()


def main():
    xml_files = sorted(ANNOT_DIR.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in {ANNOT_DIR}")
        sys.exit(1)

    pairs = []
    for xml_path in xml_files:
        img_path = IMAGES_DIR / (xml_path.stem + ".jpg")
        if img_path.exists():
            pairs.append((img_path, xml_path))

    if not pairs:
        print("No matching image-annotation pairs found.")
        sys.exit(1)

    print(f"Found {len(pairs)} image-annotation pairs.")
    print("Controls:")
    print("  Image focused:  \u2191/\u2193 = prev/next")
    print("  Ctrl+\u2191/\u2193:        prev/next (works everywhere)")
    print("  Scroll/+/-:     Zoom in/out")
    print("  Ctrl+0:         Reset zoom to fit")
    print("  A:              Toggle all annotations")
    print("  Click+drag:     Draw bounding box")
    print("  Right/Mid-drag: Pan image")
    print("  Ctrl+q = quit")
    print("Auto-saves after 500ms.\n")

    root = tk.Tk()
    root.geometry("1600x780")
    AnnotationApp(root, pairs)
    root.mainloop()


if __name__ == "__main__":
    main()
