"""S2TLD annotation helper — view and edit Pascal VOC XML annotations.

Layout: Image (left) | Editable XML (right)

Controls:
    Image focused:  ↑/↓ = prev/next image
    XML focused:    ↑/↓ = normal cursor movement
    Ctrl+↑/↓:      prev/next image (works everywhere)
    Ctrl+q:         Quit

XML edits auto-save after 500ms of inactivity.
Tracks reviewed count: an image counts as "reviewed" once you navigate away from it.
"""

import json
import sys
import tkinter as tk
from tkinter import font as tkfont
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "raw" / "S2TLD" / "JPEGImages"
ANNOT_DIR = ROOT / "data" / "raw" / "S2TLD" / "Annotations-fix"
PROGRESS_FILE = ANNOT_DIR / ".review_progress.json"

# Phase 2: 12 classes — color × shape (RGB tuples for PIL)
CLASS_COLORS = {
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "redLeft": (200, 0, 0),
    "yellowLeft": (200, 200, 0),
    "greenLeft": (0, 200, 0),
    "redForward": (255, 80, 80),
    "yellowForward": (255, 255, 80),
    "greenForward": (80, 255, 80),
    "redRight": (255, 0, 128),
    "yellowRight": (255, 255, 128),
    "greenRight": (0, 255, 128),
    "off": (128, 128, 128),
    "wait_on": (200, 200, 200),
}


def parse_annotation(xml_text: str) -> list[dict]:
    """Parse Pascal VOC XML string, return list of {name, xmin, ymin, xmax, ymax}."""
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
            })
        except ValueError:
            continue
    return objects


def draw_annotations(img: Image.Image, objects: list[dict]) -> Image.Image:
    """Draw bounding boxes and labels on a PIL image."""
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    try:
        label_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except (OSError, AttributeError):
        label_font = ImageFont.load_default()

    for i, obj in enumerate(objects):
        color = CLASS_COLORS.get(obj["name"], (255, 255, 255))
        x1, y1, x2, y2 = obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"[{i}] {obj['name']}"
        bbox = draw.textbbox((x1, y1), label, font=label_font)
        draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0), font=label_font)
    return canvas


class AnnotationApp:
    def __init__(self, root: tk.Tk, pairs: list[tuple[Path, Path]]):
        self.root = root
        self.pairs = pairs
        self.idx = 0
        self.save_job = None
        self.suppressing = False
        self.dirty = False
        self._last_text = ""
        self._current_img = None  # cache original PIL image for resize
        self._current_objects = None

        # Load review progress
        self.reviewed: set[str] = set()
        self._load_progress()

        self.root.title("S2TLD Annotation Helper")
        self.root.configure(bg="#2b2b2b")

        # --- Top bar ---
        topbar = tk.Frame(root, bg="#333333", pady=4)
        topbar.pack(fill=tk.X)

        btn_prev = tk.Button(topbar, text="◀ Prev", command=self.prev_image,
                             bg="#555", fg="white", relief=tk.FLAT, padx=8)
        btn_prev.pack(side=tk.LEFT, padx=6)

        self.nav_label = tk.Label(topbar, text="", bg="#333333", fg="white",
                                  font=("Menlo", 13))
        self.nav_label.pack(side=tk.LEFT, padx=10)

        btn_next = tk.Button(topbar, text="Next ▶", command=self.next_image,
                             bg="#555", fg="white", relief=tk.FLAT, padx=8)
        btn_next.pack(side=tk.LEFT, padx=6)

        self.progress_label = tk.Label(topbar, text="", bg="#333333", fg="#66bb6a",
                                       font=("Menlo", 12))
        self.progress_label.pack(side=tk.LEFT, padx=20)

        self.status_label = tk.Label(topbar, text="", bg="#333333", fg="#888888",
                                     font=("Menlo", 11))
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # --- Main content: image left, xml right ---
        content = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#2b2b2b",
                                 sashwidth=4, sashrelief=tk.RAISED)
        content.pack(fill=tk.BOTH, expand=True)

        # Left: image in a canvas that scales with window
        self.img_canvas = tk.Canvas(content, bg="#1e1e1e", highlightthickness=0)
        content.add(self.img_canvas, stretch="always")
        self.img_canvas.bind("<Configure>", self._on_canvas_resize)
        # Make image panel focusable and bind arrow keys
        self.img_canvas.bind("<FocusIn>", lambda e: self._update_focus_hint())
        self.img_canvas.bind("<Up>", lambda e: self.prev_image())
        self.img_canvas.bind("<Down>", lambda e: self.next_image())
        self.img_canvas.bind("<Button-1>", lambda e: self.img_canvas.focus_set())

        # Right: XML editor
        xml_frame = tk.Frame(content, bg="#2b2b2b")
        content.add(xml_frame, width=480, stretch="never")

        xml_header = tk.Label(xml_frame, text="XML Annotation (auto-saves)",
                              bg="#2b2b2b", fg="#aaaaaa", font=("Menlo", 11),
                              anchor=tk.W)
        xml_header.pack(fill=tk.X, padx=4, pady=(4, 0))

        ref_text = "Classes: red yellow green | redLeft yellowLeft greenLeft | redForward yellowForward greenForward | redRight yellowRight greenRight | off wait_on"
        ref_label = tk.Label(xml_frame, text=ref_text, bg="#2b2b2b", fg="#666666",
                             font=("Menlo", 9), anchor=tk.W, wraplength=460, justify=tk.LEFT)
        ref_label.pack(fill=tk.X, padx=4, pady=(0, 4))

        editor_frame = tk.Frame(xml_frame)
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        scrollbar = tk.Scrollbar(editor_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        mono = tkfont.Font(family="Menlo", size=12)
        self.xml_editor = tk.Text(editor_frame, bg="#1e1e1e", fg="#d4d4d4",
                                  insertbackground="white", font=mono,
                                  undo=True, wrap=tk.NONE, relief=tk.FLAT,
                                  yscrollcommand=scrollbar.set)
        self.xml_editor.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.xml_editor.yview)

        self.xml_editor.bind("<<Modified>>", self._on_text_modified)
        self.xml_editor.bind("<KeyRelease>", self._on_key_edit)
        self.xml_editor.bind("<FocusIn>", lambda e: self._update_focus_hint())

        # Global key bindings (Ctrl+arrows work everywhere)
        self.root.bind("<Control-Up>", lambda e: self.prev_image())
        self.root.bind("<Control-Down>", lambda e: self.next_image())
        self.root.bind("<Control-q>", lambda e: self._quit())

        # Focus hint
        self.focus_hint = tk.Label(topbar, text="", bg="#333333", fg="#555555",
                                   font=("Menlo", 10))
        self.focus_hint.pack(side=tk.RIGHT, padx=10)

        # Start with focus on image panel
        self.root.after(100, lambda: self.img_canvas.focus_set())

        self.load_current()

    # --- Progress tracking ---

    def _load_progress(self):
        if PROGRESS_FILE.exists():
            try:
                data = json.loads(PROGRESS_FILE.read_text())
                self.reviewed = set(data.get("reviewed", []))
                # Resume from last position
                last = data.get("last_index", 0)
                if 0 <= last < len(self.pairs):
                    self.idx = last
            except (json.JSONDecodeError, OSError):
                pass

    def _save_progress(self):
        data = {
            "reviewed": sorted(self.reviewed),
            "last_index": self.idx,
        }
        try:
            PROGRESS_FILE.write_text(json.dumps(data, indent=2))
        except OSError:
            pass

    def _mark_reviewed(self):
        _, xml_path = self.pairs[self.idx]
        self.reviewed.add(xml_path.name)

    def _update_progress_label(self):
        total = len(self.pairs)
        done = len(self.reviewed)
        remaining = total - done
        self.progress_label.config(text=f"{done}/{total} reviewed  |  {remaining} remaining")

    def _update_focus_hint(self):
        focused = self.root.focus_get()
        if focused == self.img_canvas:
            self.focus_hint.config(text="[Image] ↑↓=navigate")
        elif focused == self.xml_editor:
            self.focus_hint.config(text="[XML] Ctrl+↑↓=navigate")
        else:
            self.focus_hint.config(text="")

    # --- Text/save ---

    def _on_text_modified(self, event=None):
        if not self.xml_editor.edit_modified():
            return
        self.xml_editor.edit_modified(False)
        if self.suppressing:
            return
        self._schedule_save()
        self._refresh_image_from_editor()

    def _on_key_edit(self, event=None):
        """Backup change detector — catches edits that <<Modified>> misses."""
        if self.suppressing:
            return
        current_text = self.xml_editor.get("1.0", tk.END)
        if not hasattr(self, "_last_text") or current_text != self._last_text:
            self._last_text = current_text
            self._schedule_save()
            self._refresh_image_from_editor()

    def _schedule_save(self):
        if self.save_job:
            self.root.after_cancel(self.save_job)
        self.save_job = self.root.after(500, self._auto_save)
        self.dirty = True

    def _auto_save(self):
        self.save_job = None
        self._do_save()

    def _do_save(self):
        """Actually write to disk."""
        _, xml_path = self.pairs[self.idx]
        text = self.xml_editor.get("1.0", tk.END).rstrip("\n") + "\n"
        try:
            xml_path.write_text(text, encoding="utf-8")
            self.dirty = False
            self.status_label.config(text="Saved", fg="#66bb6a")
            self.root.after(2000, lambda: self.status_label.config(text="", fg="#888888"))
        except OSError as e:
            self.status_label.config(text=f"Save error: {e}", fg="#ef5350")

    def _refresh_image_from_editor(self):
        if self._current_img is None:
            return
        xml_text = self.xml_editor.get("1.0", tk.END)
        self._current_objects = parse_annotation(xml_text)
        self._render_image()

    # --- Image display (fits to canvas size) ---

    def _on_canvas_resize(self, event=None):
        self._render_image()

    def _render_image(self):
        if self._current_img is None:
            return
        objects = self._current_objects or []
        annotated = draw_annotations(self._current_img, objects)

        # Fit to canvas dimensions
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        img_w, img_h = annotated.size
        scale = min(cw / img_w, ch / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = annotated.resize((new_w, new_h), Image.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(resized)
        self.img_canvas.delete("all")
        # Center in canvas
        x = (cw - new_w) // 2
        y = (ch - new_h) // 2
        self.img_canvas.create_image(x, y, anchor=tk.NW, image=self._tk_img)

    def load_current(self):
        img_path, xml_path = self.pairs[self.idx]

        # Nav label with reviewed marker
        marker = " *" if xml_path.name in self.reviewed else ""
        self.nav_label.config(
            text=f"[{self.idx + 1}/{len(self.pairs)}]  {img_path.name}{marker}"
        )
        self._update_progress_label()

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

        self.suppressing = True
        self.xml_editor.delete("1.0", tk.END)
        self.xml_editor.insert("1.0", xml_text.rstrip("\n"))
        self.xml_editor.edit_modified(False)
        self.xml_editor.edit_reset()
        self._last_text = self.xml_editor.get("1.0", tk.END)
        self.dirty = False
        self.suppressing = False

        self._current_objects = parse_annotation(xml_text)
        self._render_image()
        self.status_label.config(text="", fg="#888888")

    def prev_image(self):
        self._save_now()
        self._mark_reviewed()
        self.idx = (self.idx - 1) % len(self.pairs)
        self.load_current()
        self._save_progress()

    def next_image(self):
        self._save_now()
        self._mark_reviewed()
        self.idx = (self.idx + 1) % len(self.pairs)
        self.load_current()
        self._save_progress()

    def _save_now(self):
        """Force-save current editor content immediately."""
        if self.save_job:
            self.root.after_cancel(self.save_job)
            self.save_job = None
        # Always save — don't rely on dirty flag alone
        self._do_save()

    def _quit(self):
        self._save_now()
        self._mark_reviewed()
        self._save_progress()
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
    print("  Image focused:  ↑/↓ = prev/next")
    print("  XML focused:    ↑/↓ = cursor, Ctrl+↑/↓ = prev/next")
    print("  Ctrl+q = quit")
    print("XML edits auto-save after 500ms.\n")

    root = tk.Tk()
    root.geometry("1600x780")
    AnnotationApp(root, pairs)
    root.mainloop()


if __name__ == "__main__":
    main()
