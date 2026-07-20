"""Genere assets/livecap.ico, l'icone du raccourci Bureau.

Dessine un panneau de sous-titres reprenant l'identite visuelle de l'overlay
(fond noir arrondi, ligne VO blanche, ligne traduite jaune) plutot que de
pointer vers un index d'icone systeme, dont la numerotation n'est pas un
contrat stable entre versions de Windows.

Qt sait ecrire le format ICO (QImageWriter.supportedImageFormats contient
"ico"), donc aucune dependance supplementaire n'est necessaire: PyQt6 est
deja requis par l'overlay.

Usage:
    QT_QPA_PLATFORM=offscreen .venv/Scripts/python.exe tools/make_icon.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QRectF, Qt  # noqa: E402  (must follow the platform setup)
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QPainter  # noqa: E402

# Sizes Windows actually picks between for taskbar, desktop and tooltips.
SIZES = (16, 24, 32, 48, 64, 128, 256)

PANEL_COLOR = QColor(12, 12, 12)
BORDER_COLOR = QColor(70, 70, 70)
VO_COLOR = QColor(255, 255, 255)
TRANSLATION_COLOR = QColor(255, 221, 87)

OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "livecap.ico"


def render(size: int) -> QImage:
    """Draw one square frame of the icon at `size` pixels."""
    image = QImage(size, size, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    unit = size / 32.0  # design grid: everything is expressed in 32nds
    panel = QRectF(unit * 2, unit * 7, unit * 28, unit * 18)
    radius = unit * 3

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(PANEL_COLOR)
    painter.drawRoundedRect(panel, radius, radius)

    # A hairline border keeps the panel readable against a dark wallpaper.
    if size >= 32:
        pen_width = max(1.0, unit * 0.6)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = painter.pen()
        pen.setColor(BORDER_COLOR)
        pen.setWidthF(pen_width)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawRoundedRect(panel, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)

    # Two subtitle lines: white source, yellow translation -- same colour
    # coding as the overlay itself, so the icon reads as "that black box".
    bar_height = unit * 3
    painter.setBrush(VO_COLOR)
    painter.drawRect(QRectF(unit * 5, unit * 11, unit * 22, bar_height))
    painter.setBrush(TRANSLATION_COLOR)
    painter.drawRect(QRectF(unit * 5, unit * 17, unit * 15, bar_height))

    painter.end()
    return image


def main() -> int:
    # QImage painting needs a QGuiApplication instance even offscreen.
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    assert app is not None

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames = [render(size) for size in SIZES]

    # QImage.save() writes a single frame; QIcon/QImageWriter cannot emit a
    # multi-resolution ICO. Windows scales one 256x256 frame acceptably, so
    # write the largest and let the shell downscale.
    largest = frames[-1]
    if not largest.save(str(OUTPUT), "ICO"):
        print(f"echec d'ecriture de {OUTPUT}", file=sys.stderr)
        return 1

    print(f"{OUTPUT} ecrit ({largest.width()}x{largest.height()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
