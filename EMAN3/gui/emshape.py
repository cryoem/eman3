#!/usr/bin/env python
#
# Author: Steven Ludtke, 11/01/2007 (sludtke@bcm.edu)
# Copyright (c) 2000-2006 Baylor College of Medicine
#
# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
#

"""
EMAN3 Shape system for 2-D annotations on visualization widgets.

Each shape subclass represents a specific geometric annotation type with
Python properties for parameters and a render() method for PyGfx output.
Shapes are either in data coordinates or screen coordinates (prefixed "scr").

Usage:
    shapes = {}
    shapes["crosshair"] = ShapeLine(1.0, 0.0, 0.0, 0, 100, 200, 100, line_width=2)
    shapes["label"] = ShapeLabel(0.0, 1.0, 0.0, 10, 10, "Hello", font_size=14)

    # In your widget's render loop:
    for name, shape in shapes.items():
        shape.render(renderer, d2s)
"""

import numpy as np
from math import sqrt
import pygfx as gfx


class PyGfxRenderer:
	"""Bridge between EMShape render() calls and a pygfx scene group.

	Shapes call renderer.line_strip(), renderer.points(), renderer.text(),
	and the helper creates pygfx nodes and adds them to the scene group.
	"""
	def __init__(self, scene_group):
		self._group = scene_group
		if self._group is None:
			self._group = gfx.Group()
		self._group.render_order = 10

	def line_strip(self, vertices_2d, color=(1, 1, 1), width=1.0):
		"""Draw a line strip from 2D vertices."""
		h, w = vertices_2d.shape if len(vertices_2d.shape) == 2 else (vertices_2d.size, 2)
		if h < 2:
			return
		# Convert 2D to 3D for pygfx - place in front of image at slightly positive Z
		pos = np.zeros((h, 3), dtype=np.float32)
		pos[:, :2] = vertices_2d.astype(np.float32)
		pos[:, 2] = 1.0  # In front (positive Z is closer to camera looking down +Z axis in pygfx ortho camera)
		node = gfx.Line(
			gfx.Geometry(positions=pos),
			gfx.LineThinMaterial(thickness=max(1.0, float(width)), color=(*color, 1.0)))
		self._group.add(node)

	def points(self, positions_2d, color=(1, 1, 1), size=5.0):
		"""Draw point markers from 2D positions."""
		pos = np.zeros((len(positions_2d), 3), dtype=np.float32)
		pos[:, :2] = np.asarray(positions_2d, dtype=np.float32)
		pos[:, 2] = 1.0
		colors = np.full((len(positions_2d), 4), [*color, 1.0], dtype=np.float32)
		node = gfx.Points(
			gfx.Geometry(positions=pos, colors=colors),
			gfx.PointsMaterial(size=float(size)))
		self._group.add(node)

	def text(self, x, y, txt, color=(1, 1, 1), size=12, bg=False):
		"""Draw text at screen position."""
		try:
			node = gfx.Text(str(txt), font_size=size, screen_space=True)
			node.local.position = (float(x), float(y), 0)
			node.material.color = (*color, 1.0)
			self._group.add(node)
		except Exception:
			pass


# Circle vertex cache (60 segments)
_X = np.linspace(0.0, 2.0 * np.pi, 61)
_CIRCLE_VERTS = np.column_stack([np.cos(_X[:-1]), np.sin(_X[:-1])])


def shidentity(x, y):
	"""Identity data-to-screen transform."""
	return x, y


class EMShape:
	"""Base class for all annotation shapes.

	Provides common properties (color, line_width, visible) and the interface
	that all subclasses implement.
	"""
	def __init__(self, r=0.0, g=0.0, b=0.0, line_width=1.0):
		self._r = float(r)
		self._g = float(g)
		self._b = float(b)
		self._line_width = float(line_width)
		self._visible = True

	# --- Color properties ---

	@property
	def r(self):
		return self._r

	@r.setter
	def r(self, value):
		self._r = float(value)

	@property
	def g(self):
		return self._g

	@g.setter
	def g(self, value):
		self._g = float(value)

	@property
	def b(self):
		return self._b

	@b.setter
	def b(self, value):
		self._b = float(value)

	@property
	def color(self):
		"""RGB tuple."""
		return (self._r, self._g, self._b)

	@color.setter
	def color(self, rgb):
		self._r, self._g, self._b = float(rgb[0]), float(rgb[1]), float(rgb[2])

	def set_color(self, r=None, g=None, b=None):
		"""Set color as (r,g,b) or pass a sequence."""
		if r is None:
			return
		if isinstance(r, (tuple, list, np.ndarray)):
			self._r, self._g, self._b = float(r[0]), float(r[1]), float(r[2])
		else:
			self._r, self._g, self._b = float(r), float(g), float(b)

	# --- Line width ---

	@property
	def line_width(self):
		return self._line_width

	@line_width.setter
	def line_width(self, value):
		self._line_width = float(value)

	# --- Visibility ---

	@property
	def visible(self):
		return self._visible

	@visible.setter
	def visible(self, value):
		self._visible = bool(value)

	# --- Rendering interface ---

	def render(self, renderer, d2s=shidentity):
		"""Render this shape using a PyGfx-compatible renderer.

		Args:
			renderer: A PyGfx Line/Point/Mesh rendering helper or the canvas context.
			d2s: Callable (x, y) -> (sx, sy) that maps data coords to screen coords.
		"""
		raise NotImplementedError("Subclasses must implement render()")

	def collision(self, x, y, fuzzy=False):
		"""Return True if point (x,y) is inside/on this shape."""
		raise NotImplementedError

	def control_pts(self):
		"""Return a tuple of control-point coordinates for interactive manipulation."""
		return ()

	def control_pt_min_distance(self, x, y):
		pts = self.control_pts()
		if not pts:
			return float('inf')
		dists = [sqrt((p[0] - x) ** 2 + (p[1] - y) ** 2) for p in pts]
		return min(dists)

	def __repr__(self):
		return f"<{type(self).__name__} {self.color}>"


# ─── Data-coordinate shapes ───────────────────────────────────────────────

class ShapeRect(EMShape):
	"""Axis-aligned rectangle in data coordinates.

	PARAMS: [type, R, G, B, x0, y0, x1, y1, line_width]
	"""
	def __init__(self, r, g, b, x0, y0, x1, y1, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x0 = float(x0)
		self.y0 = float(y0)
		self.x1 = float(x1)
		self.y1 = float(y1)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		p0 = d2s(self.x0, self.y0)
		p1 = (d2s(self.x1, self.y0)[0], d2s(self.x0, self.y0)[1])
		p2 = d2s(self.x1, self.y1)
		p3 = (d2s(self.x0, self.y1)[0], d2s(self.x1, self.y1)[1])
		vertices = np.array([p0, p1, p2, p3, p0], dtype=np.float32)
		renderer.line_strip(vertices, color=self.color, width=self._line_width)

	def collision(self, x, y, fuzzy=False):
		xmin, xmax = min(self.x0, self.x1), max(self.x0, self.x1)
		ymin, ymax = min(self.y0, self.y1), max(self.y0, self.y1)
		return xmin <= x <= xmax and ymin <= y <= ymax

	def control_pts(self):
		return ((self.x0, self.y0), (self.x1, self.y0),
		        (self.x1, self.y1), (self.x0, self.y1))


class ShapePoint(EMShape):
	"""A single point marker in data coordinates.

	PARAMS: [type, R, G, B, x, y, radius]
	"""
	def __init__(self, r, g, b, x, y, radius=5.0, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x = float(x)
		self.y = float(y)
		self.radius = float(radius)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		sx, sy = d2s(self.x, self.y)
		renderer.points(np.array([[sx, sy]], dtype=np.float32),
		                color=self.color, size=self.radius)

	def collision(self, x, y, fuzzy=False):
		d = sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
		return d <= self.radius

	def control_pts(self):
		return ((self.x, self.y),)


class ShapeLine(EMShape):
	"""A line segment in data coordinates.

	PARAMS: [type, R, G, B, x0, y0, x1, y1, line_width]
	"""
	def __init__(self, r, g, b, x0, y0, x1, y1, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x0 = float(x0)
		self.y0 = float(y0)
		self.x1 = float(x1)
		self.y1 = float(y1)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		p0 = np.array(d2s(self.x0, self.y0), dtype=np.float32)
		p1 = np.array(d2s(self.x1, self.y1), dtype=np.float32)
		renderer.line_strip(np.stack([p0, p1]), color=self.color, width=self._line_width)

	def collision(self, x, y, fuzzy=False):
		dx = self.x1 - self.x0
		dy = self.y1 - self.y0
		length_sq = dx * dx + dy * dy
		if length_sq == 0:
			return False
		t = max(0.0, min(1.0, ((x - self.x0) * dx + (y - self.y0) * dy) / length_sq))
		cx = self.x0 + t * dx
		cy = self.y0 + t * dy
		d = sqrt((x - cx) ** 2 + (y - cy) ** 2)
		threshold = self._line_width if fuzzy else self._line_width / 2.0
		return d <= threshold

	def control_pts(self):
		mx = (self.x0 + self.x1) / 2.0
		my = (self.y0 + self.y1) / 2.0
		return ((self.x0, self.y0), (self.x1, self.y1), (mx, my))


class ShapeCircle(EMShape):
	"""A circle in data coordinates.

	PARAMS: [type, R, G, B, cx, cy, radius, line_width]
	"""
	def __init__(self, r, g, b, cx, cy, radius, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.cx = float(cx)
		self.cy = float(cy)
		self.radius = float(radius)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		# Scale circle vertices by data-space unit size
		v0 = np.array(d2s(self.cx, self.cy), dtype=np.float32)
		v1 = np.array(d2s(self.cx + 1.0, self.cy + 1.0), dtype=np.float32)
		sc = (v1[0] - v0[0])  # screen pixels per data unit

		circle_verts = _CIRCLE_VERTS * (self.radius * sc) + v0
		renderer.line_strip(np.concatenate([circle_verts, circle_verts[:1]], axis=0),
		                    color=self.color, width=self._line_width)

	def collision(self, x, y, fuzzy=False):
		d = sqrt((self.cx - x) ** 2 + (self.cy - y) ** 2)
		return abs(d - self.radius) < self._line_width / 2.0

	def control_pts(self):
		return ((self.cx, self.cy),)


class ShapeEllipse(EMShape):
	"""An ellipse in data coordinates.

	PARAMS: [type, R, G, B, cx, cy, r1, r2, angle_deg, line_width]
	"""
	def __init__(self, r, g, b, cx, cy, r1, r2, angle_deg=0.0, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.cx = float(cx)
		self.cy = float(cy)
		self.r1 = float(r1)
		self.r2 = float(r2)
		self.angle_deg = float(angle_deg)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		v0 = np.array(d2s(self.cx, self.cy), dtype=np.float32)
		v1 = np.array(d2s(self.cx + 1.0, self.cy + 1.0), dtype=np.float32)
		sc = v1[0] - v0[0]

		r1_sc = self.r1 * sc
		r2_sc = self.r2 * sc
		angle_rad = np.deg2rad(self.angle_deg)

		cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

		base = _CIRCLE_VERTS.copy()
		base[:, 0] *= r1_sc
		base[:, 1] *= r2_sc

		rotated = base @ np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
		rotated += v0

		renderer.line_strip(np.concatenate([rotated, rotated[:1]], axis=0),
		                    color=self.color, width=self._line_width)

	def collision(self, x, y, fuzzy=False):
		dx = x - self.cx
		dy = y - self.cy
		angle_rad = np.deg2rad(-self.angle_deg)
		cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
		lx = dx * cos_a + dy * sin_a
		ly = -dx * sin_a + dy * cos_a
		if self.r1 > 0 and self.r2 > 0:
			val = (lx / self.r1) ** 2 + (ly / self.r2) ** 2
			return abs(val - 1.0) < 0.1
		return False

	def control_pts(self):
		return ((self.cx, self.cy),)


class ShapeRectLine(EMShape):
	"""A rectangle defined by an axis (two points) and a width, with a center line.

	PARAMS: [type, R, G, B, x0, y0, x1, y1, box_width, line_width]
	"""
	def __init__(self, r, g, b, x0, y0, x1, y1, box_width, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x0 = float(x0)
		self.y0 = float(y0)
		self.x1 = float(x1)
		self.y1 = float(y1)
		self.box_width = float(box_width)

	@property
	def centroid(self):
		return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

	def _corners(self):
		dx = self.x1 - self.x0
		dy = self.y1 - self.y0
		mag = sqrt(dx * dx + dy * dy)
		if mag == 0:
			return ((self.x0, self.y0),) * 4
		wx, wy = -dy / mag, dx / mag
		hw = self.box_width / 2.0
		return (
			(self.x0 - wx * hw, self.y0 - wy * hw),
			(self.x0 + wx * hw, self.y0 + wy * hw),
			(self.x1 + wx * hw, self.y1 + wy * hw),
			(self.x1 - wx * hw, self.y1 - wy * hw),
		)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		corners = self._corners()
		vertices = np.array([d2s(*c) for c in [corners[0], corners[1], corners[2], corners[3], corners[0]]], dtype=np.float32)
		renderer.line_strip(vertices, color=self.color, width=self._line_width)
		p0 = np.array(d2s(self.x0, self.y0), dtype=np.float32)
		p1 = np.array(d2s(self.x1, self.y1), dtype=np.float32)
		renderer.line_strip(np.stack([p0, p1]), color=self.color, width=self._line_width)

	def collision(self, x, y, fuzzy=False):
		cx, cy = self.centroid
		dx = self.x1 - self.x0
		dy = self.y1 - self.y0
		length = sqrt(dx * dx + dy * dy)
		if length == 0:
			return False
		lx, ly = dx / length, dy / length
		wx, wy = ly, -lx
		tx, ty = x - cx, y - cy
		w_proj = tx * wx + ty * wy
		l_proj = tx * lx + ty * ly
		hw = self.box_width / 2.0
		if abs(w_proj) > hw:
			return False
		if fuzzy:
			return abs(l_proj) <= 5.0 * length / 8.0
		return abs(l_proj) <= length / 2.0

	def control_pts(self):
		cx, cy = self.centroid
		return ((self.x0, self.y0), (self.x1, self.y1), (cx, cy))


class ShapeRectPoint(EMShape):
	"""A rectangle with a point marker at the center.

	PARAMS: [type, R, G, B, x0, y0, x1, y1, line_width]
	"""
	def __init__(self, r, g, b, x0, y0, x1, y1, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x0 = float(x0)
		self.y0 = float(y0)
		self.x1 = float(x1)
		self.y1 = float(y1)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		p0 = d2s(self.x0, self.y0)
		p1 = (d2s(self.x1, self.y0)[0], d2s(self.x0, self.y0)[1])
		p2 = d2s(self.x1, self.y1)
		p3 = (d2s(self.x0, self.y1)[0], d2s(self.x1, self.y1)[1])
		vertices = np.array([p0, p1, p2, p3, p0], dtype=np.float32)
		renderer.line_strip(vertices, color=self.color, width=self._line_width)
		mx = (p0[0] + d2s(self.x1, self.y1)[0]) / 2.0
		my = (p0[1] + d2s(self.x1, self.y1)[1]) / 2.0
		renderer.points(np.array([[mx, my]], dtype=np.float32), color=self.color, size=self._line_width)


class ShapeRCirclePoint(EMShape):
	"""An inscribed circle in the rectangle defined by (x0,y0)-(x1,y1), with a center point.

	PARAMS: [type, R, G, B, x0, y0, x1, y1, line_width]
	"""
	def __init__(self, r, g, b, x0, y0, x1, y1, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x0 = float(x0)
		self.y0 = float(y0)
		self.x1 = float(x1)
		self.y1 = float(y1)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		v0 = np.array(d2s(self.x0, self.y0), dtype=np.float32)
		v1 = np.array(d2s(self.x1, self.y1), dtype=np.float32)
		cx = (v0[0] + v1[0]) / 2.0
		cy = (v0[1] + v1[1]) / 2.0
		rx = abs(v1[0] - v0[0]) / 2.0
		ry = abs(v1[1] - v0[1]) / 2.0

		ellipse_verts = _CIRCLE_VERTS.copy()
		ellipse_verts[:, 0] *= rx
		ellipse_verts[:, 1] *= ry
		ellipse_verts += [cx, cy]

		renderer.line_strip(np.concatenate([ellipse_verts, ellipse_verts[:1]], axis=0),
		                    color=self.color, width=self._line_width)
		renderer.points(np.array([[cx, cy]], dtype=np.float32), color=self.color, size=self._line_width)


class ShapeRCircle(EMShape):
	"""An inscribed circle in the rectangle defined by (x0,y0)-(x1,y1).

	PARAMS: [type, R, G, B, x0, y0, x1, y1, line_width]
	"""
	def __init__(self, r, g, b, x0, y0, x1, y1, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x0 = float(x0)
		self.y0 = float(y0)
		self.x1 = float(x1)
		self.y1 = float(y1)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		v0 = np.array(d2s(self.x0, self.y0), dtype=np.float32)
		v1 = np.array(d2s(self.x1, self.y1), dtype=np.float32)
		cx = (v0[0] + v1[0]) / 2.0
		cy = (v0[1] + v1[1]) / 2.0
		rx = abs(v1[0] - v0[0]) / 2.0
		ry = abs(v1[1] - v0[1]) / 2.0

		ellipse_verts = _CIRCLE_VERTS.copy()
		ellipse_verts[:, 0] *= rx
		ellipse_verts[:, 1] *= ry
		ellipse_verts += [cx, cy]

		renderer.line_strip(np.concatenate([ellipse_verts, ellipse_verts[:1]], axis=0),
		                    color=self.color, width=self._line_width)


class ShapeLabel(EMShape):
	"""A text label in data coordinates.

	PARAMS: [type, R, G, B, x, y, text, font_size, line_width (negative for background)]
	"""
	def __init__(self, r, g, b, x, y, text, font_size=12, bg=False):
		super().__init__(r, g, b)
		self.x = float(x)
		self.y = float(y)
		self.text = str(text)
		self.font_size = int(font_size)
		self._bg = bool(bg)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		sx, sy = d2s(self.x, self.y)
		renderer.text(sx, sy, self.text, color=self.color, size=self.font_size, bg=self._bg)

	def collision(self, x, y, fuzzy=False):
		return False


class ShapeMask(EMShape):
	"""A quadrilateral mask drawn as a triangle strip (4 corner pairs = 8 params after color).

	PARAMS: [type, R, G, B, xi0, yi0, xo0, yo0, ..., xi3, yi3, xo3, yo3]
	Simplified: 4 control points defining the shape.
	"""
	def __init__(self, r, g, b, *pts, line_width=1.0):
		"""*pts can be 4 (x,y) tuples or a flat list of 8 numbers."""
		super().__init__(r, g, b, line_width)
		if len(pts) == 1 and isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 8:
			self.points = [(pts[0][i], pts[0][i + 1]) for i in range(0, min(len(pts[0]), 8), 2)]
		else:
			self.points = list(pts)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		vertices = np.array([d2s(*p) for p in self.points], dtype=np.float32)
		renderer.triangle_strip(vertices, color=self.color, width=self._line_width)

	def control_pts(self):
		return tuple(self.points)


class ShapeLineMask(EMShape):
	"""A closed polygon outline (4 points minimum).

	PARAMS: [type, R, G, B, x0, y0, x1, y1, x2, y2, x3, y3, line_width]
	"""
	def __init__(self, r, g, b, *pts, line_width=1.0):
		if len(pts) == 1 and isinstance(pts[0], (list, tuple)) and len(pts[0]) >= 8:
			self.points = [(pts[0][i], pts[0][i + 1]) for i in range(0, min(len(pts[0]), 12), 2)]
		else:
			self.points = list(pts)
		super().__init__(r, g, b, line_width)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		vertices = np.array([d2s(*p) for p in self.points + [self.points[0]]], dtype=np.float32)
		renderer.line_strip(vertices, color=self.color, width=self._line_width)

	def control_pts(self):
		return tuple(self.points)


# ─── Screen-coordinate shapes ──────────────────────────────────────────────

class ShapeScrRect(EMShape):
	"""Axis-aligned rectangle in screen coordinates."""
	def __init__(self, r, g, b, x0, y0, x1, y1, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x0 = int(round(float(x0)))
		self.y0 = int(round(float(y0)))
		self.x1 = int(round(float(x1)))
		self.y1 = int(round(float(y1)))

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		vertices = np.array([
			[self.x0, self.y0], [self.x1, self.y0],
			[self.x1, self.y1], [self.x0, self.y1], [self.x0, self.y0]
		], dtype=np.float32)
		renderer.line_strip(vertices, color=self.color, width=self._line_width)

	def collision(self, x, y, fuzzy=False):
		xmin, xmax = min(self.x0, self.x1), max(self.x0, self.x1)
		ymin, ymax = min(self.y0, self.y1), max(self.y0, self.y1)
		return xmin <= x <= xmax and ymin <= y <= ymax


class ShapeScrLine(EMShape):
	"""A line segment in screen coordinates."""
	def __init__(self, r, g, b, x0, y0, x1, y1, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.x0 = int(round(float(x0)))
		self.y0 = int(round(float(y0)))
		self.x1 = int(round(float(x1)))
		self.y1 = int(round(float(y1)))

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		vertices = np.array([[self.x0, self.y0], [self.x1, self.y1]], dtype=np.float32)
		renderer.line_strip(vertices, color=self.color, width=self._line_width)

	def control_pts(self):
		mx = (self.x0 + self.x1) / 2.0
		my = (self.y0 + self.y1) / 2.0
		return ((self.x0, self.y0), (self.x1, self.y1), (mx, my))


class ShapeScrLabel(EMShape):
	"""A text label in screen coordinates."""
	def __init__(self, r, g, b, x, y, text, font_size=12, bg=False):
		super().__init__(r, g, b)
		self.x = int(round(float(x)))
		self.y = int(round(float(y)))
		self.text = str(text)
		self.font_size = int(font_size)
		self._bg = bool(bg)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		renderer.text(self.x, self.y, self.text, color=self.color, size=self.font_size, bg=self._bg)


class ShapeScrCircle(EMShape):
	"""A circle in screen coordinates."""
	def __init__(self, r, g, b, cx, cy, radius, line_width=1.0):
		super().__init__(r, g, b, line_width)
		self.cx = int(round(float(cx)))
		self.cy = int(round(float(cy)))
		self.radius = float(radius)

	def render(self, renderer, d2s=shidentity):
		if not self._visible:
			return
		circle_verts = _CIRCLE_VERTS * self.radius + [self.cx, self.cy]
		vertices = np.concatenate([circle_verts, circle_verts[:1]], axis=0).astype(np.float32)
		renderer.line_strip(vertices, color=self.color, width=self._line_width)

	def collision(self, x, y, fuzzy=False):
		d = sqrt((self.cx - x) ** 2 + (self.cy - y) ** 2)
		return abs(d - self.radius) < self._line_width / 2.0


# ─── Compatibility alias: "hidden" shape that renders nothing ──────────────

class ShapeHidden(EMShape):
	"""A hidden shape that is never rendered."""
	def __init__(self, *args, **kwargs):
		super().__init__()
		self._visible = False

	def render(self, renderer, d2s=shidentity):
		pass


# ─── Legacy list-based adapter ─────────────────────────────────────────────

_SHAPE_TYPE_MAP = {
	"rect": ShapeRect,
	"rectpoint": ShapeRectPoint,
	"rectline": ShapeRectLine,
	"rcircle": ShapeRCircle,
	"rcirclepoint": ShapeRCirclePoint,
	"line": ShapeLine,
	"label": ShapeLabel,
	"circle": ShapeCircle,
	"ellipse": ShapeEllipse,
	"scrrect": ShapeScrRect,
	"scrline": ShapeScrLine,
	"scrlabel": ShapeScrLabel,
	"scrcircle": ShapeScrCircle,
	"point": ShapePoint,
	"mask": ShapeMask,
	"linemask": ShapeLineMask,
	"hidden": ShapeHidden,
}


def legacy_list_to_shape(shape_list):
	"""Convert a legacy list/tuple shape representation to a shape instance.

	Legacy format: [type_name, R, G, B, ...type-specific params...]

	This allows loading existing shape dictionaries that use the old list format.
	"""
	if not shape_list or not isinstance(shape_list, (list, tuple)):
		return ShapeHidden()

	stype = shape_list[0]
	clz = _SHAPE_TYPE_MAP.get(stype)
	if clz is None:
		return ShapeHidden()

	try:
		if stype == "circle":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			           shape_list[4], shape_list[5], shape_list[6],
			           line_width=shape_list[7] if len(shape_list) > 7 else 1.0)
		elif stype == "ellipse":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5], shape_list[6], shape_list[7],
			            shape_list[8] if len(shape_list) > 8 else 0.0,
			            line_width=shape_list[9] if len(shape_list) > 9 else 1.0)
		elif stype == "line":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5], shape_list[6], shape_list[7],
			            line_width=shape_list[8] if len(shape_list) > 8 else 1.0)
		elif stype == "point":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5],
			            shape_list[6] if len(shape_list) > 6 else 5.0)
		elif stype == "rect" or stype == "rectpoint":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5], shape_list[6], shape_list[7],
			            line_width=shape_list[8] if len(shape_list) > 8 else 1.0)
		elif stype == "rectline":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5], shape_list[6], shape_list[7],
			            shape_list[8],
			            line_width=shape_list[9] if len(shape_list) > 9 else 1.0)
		elif stype == "rcircle" or stype == "rcirclepoint":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5], shape_list[6], shape_list[7],
			            line_width=shape_list[8] if len(shape_list) > 8 else 1.0)
		elif stype == "label" or stype == "scrlabel":
			bg = False
			if len(shape_list) > 8 and shape_list[8] < 0:
				bg = True
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5],
			            shape_list[6],
			            shape_list[7] if len(shape_list) > 7 else 12,
			            bg=bg)
		elif stype == "scrrect" or stype == "scrline":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5], shape_list[6], shape_list[7],
			            line_width=shape_list[8] if len(shape_list) > 8 else 1.0)
		elif stype == "scrcircle":
			return clz(shape_list[1], shape_list[2], shape_list[3],
			            shape_list[4], shape_list[5],
			            shape_list[6],
			            line_width=shape_list[7] if len(shape_list) > 7 else 1.0)
		elif stype == "mask":
			flat = list(shape_list[1:])
			return clz(shape_list[1], shape_list[2], shape_list[3], flat)
		elif stype == "linemask":
			flat = list(shape_list[1:13])
			return clz(shape_list[1], shape_list[2], shape_list[3], flat,
			            line_width=shape_list[12] if len(shape_list) > 12 else 1.0)
		else:
			return ShapeHidden()
	except (IndexError, ValueError):
		return ShapeHidden()


# ─── Shape container ──────────────────────────────────────────────────────

class EMShapeDict(dict):
	"""Dictionary-like container for shapes with collision detection helpers."""

	def collisions(self, x, y, fuzzy=False):
		"""Return a list of keys for shapes that contain point (x,y)."""
		return [k for k in self if self[k].collision(x, y, fuzzy)]

	def closest_collision(self, x, y, fuzzy=False):
		"""Find the colliding shape with the nearest control point to (x,y).

		Returns the key or None.
		"""
		keys = self.collisions(x, y, fuzzy)
		if not keys:
			return None

		best_key = None
		best_dist = float('inf')
		for k in keys:
			d = self[k].control_pt_min_distance(x, y)
			if d < best_dist:
				best_dist = d
				best_key = k
		return best_key
