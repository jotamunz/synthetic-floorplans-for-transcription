"""A record drawn as the image a model reads: the image follows from the record and nothing else.

Every wall, outside or inside, is a black bar of one thickness along its segment, reaching half a
thickness past its ends so that walls meeting in a corner close it. A door or outside door is a gap in
its wall, closed across at each jamb; a window the same gap outlined, with a line down its middle.

A generated record's segments are all horizontal or vertical. A predicted one can hold a tilted segment,
which is drawn the same way, turned along it, so that the image shows the record as it is.
"""

import math

import numpy as np
import skia

#: The thickness of a wall, and of the lines of doors and windows, in pixels of a 512 render.
WALL = 6.0
LINE = 1.0


def draw(record: dict, size: int) -> np.ndarray:
    """`(size, size)` uint8, white background, row index = y."""
    wall, line = WALL * size / 512, LINE * size / 512
    surface = skia.Surface(size, size)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorWHITE)
    black = skia.Paint(Color=skia.ColorBLACK, AntiAlias=True)
    white = skia.Paint(Color=skia.ColorWHITE, AntiAlias=False)
    stroke = skia.Paint(Color=skia.ColorBLACK, Style=skia.Paint.kStroke_Style, StrokeWidth=line, AntiAlias=True)

    def bar(start, end, width: float, reach: float) -> skia.Rect | skia.Path:
        """The rectangle `width` across along a segment, `reach` past its ends: a `Rect` for a horizontal or
        vertical segment, a `Path` turned along a tilted one."""
        (x0, y0), (x1, y1) = sorted([tuple(start), tuple(end)])
        if y0 == y1:
            return skia.Rect.MakeLTRB(x0 - reach, y0 - width / 2, x1 + reach, y0 + width / 2)
        if x0 == x1:
            return skia.Rect.MakeLTRB(x0 - width / 2, y0 - reach, x0 + width / 2, y1 + reach)
        length = math.hypot(x1 - x0, y1 - y0)
        along_x, along_y = (x1 - x0) / length, (y1 - y0) / length
        side_x, side_y = -along_y * width / 2, along_x * width / 2
        first_x, first_y = x0 - along_x * reach, y0 - along_y * reach
        last_x, last_y = x1 + along_x * reach, y1 + along_y * reach
        path = skia.Path()
        path.moveTo(first_x + side_x, first_y + side_y)
        path.lineTo(last_x + side_x, last_y + side_y)
        path.lineTo(last_x - side_x, last_y - side_y)
        path.lineTo(first_x - side_x, first_y - side_y)
        path.close()
        return path

    def fill(shape: skia.Rect | skia.Path, paint: skia.Paint) -> None:
        if isinstance(shape, skia.Rect):
            canvas.drawRect(shape, paint)
        else:
            canvas.drawPath(shape, paint)

    def across(x0: float, y0: float, x1: float, y1: float) -> tuple:
        """Half a wall's thickness across a segment, the way a jamb closes it."""
        if y0 == y1:
            return 0, wall / 2
        if x0 == x1:
            return wall / 2, 0
        length = math.hypot(x1 - x0, y1 - y0)
        return -(y1 - y0) / length * wall / 2, (x1 - x0) / length * wall / 2

    for node in record["nodes"]:
        if node["class"].endswith("wall"):
            fill(bar(*node["points"], wall, wall / 2), black)
    for node in record["nodes"]:
        if node["class"].endswith("wall"):
            continue
        gap = bar(*node["points"], wall, 0)
        fill(bar(*node["points"], wall + 2, 0), white)
        (x0, y0), (x1, y1) = node["points"]
        if node["class"] == "window":
            fill(gap, stroke)
            canvas.drawLine(x0, y0, x1, y1, stroke)
        else:
            dx, dy = across(x0, y0, x1, y1)
            for x, y in ((x0, y0), (x1, y1)):
                canvas.drawLine(x - dx, y - dy, x + dx, y + dy, stroke)
    image = surface.makeImageSnapshot().toarray(colorType=skia.ColorType.kGray_8_ColorType)
    return np.asarray(image)[:, :, 0] if image.ndim == 3 else np.asarray(image)
