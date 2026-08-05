#!/usr/bin/env python
#
# EMAN3 3D Plot Widget - PyGfx rendering with PySide6
# Ported from EMAN2 qtgui/emplot3d.py

import weakref

import pygfx as gfx
from rendercanvas.qt import QRenderWidget

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor as QtGuiQColor
from EMAN3.gui.valslider import ValSlider, ValBox
import pylinalg as la
import numpy as np

# Color palette - black first for publication style plotting
COLORS = [
	"#000000",  # black
	"#4169E1",  # royal blue
	"#DC143C",  # crimson
	"#228B22",  # forest green
	"#00CED1",  # dark turquoise
	"#9400D3",  # dark violet
	"#FFD700",  # gold
	"#808080",  # grey
]


def _hex_to_rgba(hex_color):
	"""Convert hex color to (r,g,b,a) tuple of floats."""
	hex_color = hex_color.lstrip("#")
	r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
	return (r / 255.0, g / 255.0, b / 255.0, 1.0)


class _EMCanvas3D(QRenderWidget):
	"""QRenderWidget subclass that forwards mouse/keyboard events to parent."""

	def __init__(self, parent_widget, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._parent = parent_widget
		self.setMouseTracking(True)

	def mousePressEvent(self, event):
		self._parent._on_mouse_press(event)
		super().mousePressEvent(event)

	def mouseMoveEvent(self, event):
		self._parent._on_mouse_move(event)
		super().mouseMoveEvent(event)

	def mouseReleaseEvent(self, event):
		self._parent._on_mouse_release(event)
		super().mouseReleaseEvent(event)

	def wheelEvent(self, event):
		self._parent._on_wheel(event)
		super().wheelEvent(event)

	def keyPressEvent(self, event):
		self._parent._on_key_press(event)
		super().keyPressEvent(event)


class EMPlot3DWidget(QtWidgets.QWidget):
	"""3D scatter plot widget using PyGfx for rendering."""

	selected_sg = QtCore.Signal()
	mousedown = QtCore.Signal(object, tuple)
	mousedrag = QtCore.Signal(object, tuple)
	mouseup = QtCore.Signal(object, tuple)
	keypress = QtCore.Signal(object)

	def __init__(self, parent=None):
		super().__init__(parent)

		self.setFocusPolicy(Qt.StrongFocus)
		self.setMouseTracking(True)
		self.setMinimumSize(200, 150)

		# Data
		self.data = {}
		self.visibility = {}
		self.axes = {}        # (x_idx, y_idx, z_idx)
		self.pparm = {}
		self.comments = {}
		self.column_labels = {}

		# Display params
		self.xlimits = None   # (xmin, xmax) in data coords
		self.ylimits = None   # (ymin, ymax)
		self.zlimits = None   # (zmin, zmax)
		self.plottitle = ""
		self.xaxis_label = ""
		self.yaxis_label = ""
		self.zaxis_label = ""

		# Selection
		self.selectpoints = True
		self.selected = []
		self.mouseemit = False

		# Mouse state
		self._left_drag_active = False
		self._right_drag_active = False
		self._right_drag_start = None

		# PyGfx setup
		self._canvas = None
		self._renderer = None
		self._scene = gfx.Scene()
		self._camera = None
		self._controller = None
		self._axis_labels = []
		self._rulers = {}       # 12 gfx.Ruler objects by edge name
		self._data_groups = {}
		self._dirty = True
		self._redraw = False
		self._rebuilding = False

		self._setup_gfx()

		# Inspector
		self.inspector = None

		self.resize(640, 480)

		# Defer scene init to after first paint (renderer needs valid size)
		QtCore.QTimer.singleShot(0, self._init_scene)

	def _setup_gfx(self):
		"""Initialize PyGfx rendering pipeline."""
		self._canvas = _EMCanvas3D(parent_widget=self, parent=self)

		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self._canvas)

		self._renderer = gfx.WgpuRenderer(self._canvas)

		# White background
		self._scene.add(gfx.Background.from_color("#ffffff"))

		# Points marker material placeholder
		_dummy_pts = np.array([[0, 0, 0]], dtype=np.float32)
		self._point_placeholder = gfx.Points(
			gfx.Geometry(positions=_dummy_pts),
			gfx.PointsMarkerMaterial(size=1, marker='circle', color=(0, 0, 0, 0)))
		self._point_placeholder.visible = False
		self._scene.add(self._point_placeholder)

		# Line material placeholder
		_dummy_verts = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
		self._line_placeholder = gfx.Line(
			gfx.Geometry(positions=_dummy_verts),
			gfx.LineMaterial(thickness=1, color=(0, 0, 0, 0)))
		self._line_placeholder.visible = False
		self._scene.add(self._line_placeholder)

		# Text placeholder
		try:
			_dummy_text = gfx.Text("Init", font_size=12, render_order=999)
			_dummy_text.material.color = (0, 0, 0, 0)
			_dummy_text.local.position = (0, 0, 0)
			self._scene.add(_dummy_text)
		except Exception:
			pass

	def _data_to_world(self, x, y, z):
		"""Convert data coordinates to world space (-1 to +1 cube).

		Returns (wx, wy, wz) in float32.
		"""
		if self.xlimits is None or self.ylimits is None or self.zlimits is None:
			return x, y, z

		xmin, xmax = self.xlimits
		ymin, ymax = self.ylimits
		zmin, zmax = self.zlimits

		# Linear map: data -> [-1, +1], output as float32 for pygfx buffers
		wx = np.where(xmax != xmin, 2.0 * (x - xmin) / (xmax - xmin) - 1.0, 0.0).astype(np.float32)
		wy = np.where(ymax != ymin, 2.0 * (y - ymin) / (ymax - ymin) - 1.0, 0.0).astype(np.float32)
		wz = np.where(zmax != zmin, 2.0 * (z - zmin) / (zmax - zmin) - 1.0, 0.0).astype(np.float32)
		return wx, wy, wz

	def _scalar_to_world(self, val, limits):
		"""Convert a scalar data value to world coordinate (-1 to +1)."""
		if limits is None:
			return float(val)
		mn, mx = limits
		if mx == mn:
			return 0.0
		return 2.0 * (float(val) - float(mn)) / (float(mx) - float(mn)) - 1.0

	def _request_render(self):
		"""Request a render frame."""
		if self._canvas is None:
			return
		if self._dirty or self._redraw:
			self._canvas.request_draw(self._render_callback)

	def _render_callback(self):
		"""Render frame callback."""
		try:
			self._redraw = False

			if self._dirty and not self._rebuilding:
				self._rebuilding = True
				self._dirty = False
				QtCore.QTimer.singleShot(0, self._rebuild_data_groups)

			self._render_plot()

			if self._dirty or self._redraw:
				self._canvas.request_draw(self._render_callback)
		except Exception as e:
			print(f"3D Plot render error: {e}")
			import traceback
			traceback.print_exc()

	def _init_scene(self):
		"""One-time scene graph initialization after renderer is ready."""
		try:
			w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
			if w == 0:
				w, h = 640, 480

			self._camera = gfx.PerspectiveCamera(45.0)
			# Position camera so it frames the [-1,+1] cube
			import numpy as np
			self._camera.world.position = (1.8, 1.2, 1.8)
			self._camera.look_at(np.array([0, 0, 0]))

			# Orbit controller for interactive rotation/zoom
			self._controller = gfx.OrbitController()
			self._controller.target = (0, 0, 0)
			self._controller.add_camera(self._camera)
			# Register controller with renderer (for event handling)
			self._controller.register_events(self._renderer)

			# Data groups
			for key in self.data:
				group = gfx.Group()
				group.name = f"data-{key}"
				self._scene.add(group)
				self._data_groups[key] = group

			# 12 rulers for the cube edges (edge lines created inline)
			self._create_rulers(w, h)

			# Populate initial data
			self._rebuild_data_groups()
		except Exception as e:
			print(f"Scene init error: {e}")
			import traceback
			traceback.print_exc()

		self._dirty = True
		self._redraw = True
		self._request_render()

	def _create_rulers(self, w, h):
		"""Create gfx.Ruler objects along each cube edge for native tick marks and labels.

		12 Rulers total. Ticks shown on all edges. Text labels only on the 3 primary
		visible axes (x_bottom, y_left, z_front_left).
		"""
		labeled = {"x_bottom", "y_left_front", "z_front_left"}

		edges = [
			("x_bottom",      (-1,-1,-1), ( 1,-1,-1)),   # bottom-front horizontal
			("x_top_front",   (-1, 1,-1), ( 1, 1,-1)),   # top-front horizontal
			("y_left_front",  (-1,-1,-1), (-1, 1,-1)),   # front-left vertical
			("y_right_front", ( 1,-1,-1), ( 1, 1,-1)),   # front-right vertical
			("z_front_left",  (-1,-1,-1), (-1,-1, 1)),   # bottom-left depth
			("z_front_right", ( 1,-1,-1), ( 1,-1, 1)),   # bottom-right depth
			("x_bottom_back", (-1,-1, 1), ( 1,-1, 1)),   # back-bottom horizontal
			("x_top_back",    (-1, 1, 1), ( 1, 1,  1)),  # back-top horizontal
			("y_left_back",   (-1,-1, 1), (-1, 1, 1)),   # back-left vertical
			("y_right_back",  ( 1,-1, 1), ( 1, 1,  1)),  # back-right vertical
			("z_top_left",    (-1, 1,-1), (-1, 1, 1)),   # top-left depth
			("z_top_right",   ( 1, 1,-1), ( 1, 1,  1)),  # top-right depth
		]

		for name, start, end in edges:
			ruler = gfx.Ruler(tick_side="right",
					min_tick_distance=35, color=(0, 0, 0, 1.0))
			ruler.start_pos = start
			ruler.end_pos = end
			ruler.start_value = -1.0
			# Hide text labels on back/non-primary edges; keep ticks visible
			if name not in labeled:
				try:
					ruler.text.material.opacity = 0
				except Exception:
					pass
			self._scene.add(ruler)
			self._rulers[name] = ruler

		# Manual tick labels for Z axis (Ruler text faces camera, unreadable on depth edges)
		self._z_tick_texts = []
		for i in range(5):
			w_val = -1.0 + i * 0.5  # world position along Z
			t = gfx.Text("", font_size=12, screen_space=True, render_order=998)
			t.material.color = (0, 0, 0, 1.0)
			t.local.position = (-1.08, -1.08, w_val)  # Offset from Z edge
			self._scene.add(t)
			self._z_tick_texts.append(t)

	def _update_ruler_values(self):
		"""Set ruler start_value so tick labels show data-space coordinates."""
		xlims = self.xlimits if self.xlimits else (-1.0, 1.0)
		ylims = self.ylimits if self.ylimits else (-1.0, 1.0)
		zlims = self.zlimits if self.zlimits else (-1.0, 1.0)

		xmin, xmax = xlims
		ymin, ymax = ylims
		zmin, zmax = zlims

		x_axes = ("x_bottom", "x_top_front", "x_bottom_back", "x_top_back")
		y_axes = ("y_left_front", "y_right_front", "y_left_back", "y_right_back")
		z_axes = ("z_front_left", "z_front_right", "z_top_left", "z_top_right")

		for k in x_axes:
			r = self._rulers.get(k)
			if r:
				r.start_value = xmin
		for k in y_axes:
			r = self._rulers.get(k)
			if r:
				r.start_value = ymin
		for k in z_axes:
			r = self._rulers.get(k)
			if r:
				r.start_value = zmin

		# Update manual Z tick label texts
		if hasattr(self, '_z_tick_texts') and zlims:
			for i, t in enumerate(self._z_tick_texts):
				w_val = -1.0 + i * 0.5
				d_val = zmin + (w_val + 1.0) / 2.0 * (zmax - zmin)
				t.value = f"{d_val:.4g}"

	def _create_axis_labels(self):
		"""Create X, Y, Z axis labels anchored to world positions that orbit with the view.

		Positioned halfway along the three rulers extending from the Xmin,Ymin,Zmin corner,
		with text running parallel to each ruler. Slightly offset outward from the ruler.
		"""
		# Remove old axis labels from scene first
		for old_label in self._axis_labels:
			try:
				self._scene.remove(old_label)
			except Exception:
				pass
		self._axis_labels = []
		label_color = (0, 0, 0, 1.0)

		# X label: below x_bottom ruler edge, text runs left-right
		if self.xaxis_label:
			xlbl = gfx.Text(self.xaxis_label, font_size=0.18, render_order=999)
			xlbl.material.color = label_color
			xlbl.local.position = (0, -1.18, -1)
			self._scene.add(xlbl)
			self._axis_labels.append(xlbl)

		# Y label: left of y_left ruler edge, text runs bottom-to-top (rotate +90 around Z)
		if self.yaxis_label:
			ylbl = gfx.Text(self.yaxis_label, font_size=0.18, render_order=999)
			ylbl.material.color = label_color
			ylbl.local.position = (-1.18, 0, -1)
			ylbl.local.rotation = la.quat_from_euler((0, 0, np.radians(90)))
			self._scene.add(ylbl)
			self._axis_labels.append(ylbl)

		# Z label: offset from z_front_left ruler edge, text runs front-to-back (rotate -90 around X)
		if self.zaxis_label:
			zlbl = gfx.Text(self.zaxis_label, font_size=0.18, render_order=999)
			zlbl.material.color = label_color
			zlbl.local.position = (-1.18, -1.0, 0)
			zlbl.local.rotation = la.quat_from_euler((np.radians(180), np.radians(270), np.radians(90)))
			self._scene.add(zlbl)
			self._axis_labels.append(zlbl)

	def _render_plot(self):
		"""Position rulers, labels, and render."""
		if self._camera is None:
			return

		renderer = self._renderer
		camera = self._camera
		w, h = renderer.logical_size[0], renderer.logical_size[1]

		if w > 0 and h > 0:
			camera.set_view_size(w, h)

		# Update ruler tick values to data coordinates
		self._update_ruler_values()

		# Call update on each ruler (computes tick spacing/placement)
		for r in self._rulers.values():
			r.update(camera, (w, h))

		# Create axis labels if not already present and labels are set
		if not self._axis_labels:
			self._create_axis_labels()

		renderer.render(self._scene, camera)

	def _rebuild_data_groups(self):
		"""Rebuild data groups - called outside render callback."""
		try:
			keys = list(self.data.keys())
			old_keys = list(self._data_groups.keys())
			all_keys = list(dict.fromkeys(keys + old_keys))

			for key in all_keys:
				is_new = key not in old_keys
				is_removed = key not in keys
				visible = self.visibility.get(key, True)

				if is_new:
					group = gfx.Group()
					group.name = f"data-{key}"
					self._scene.add(group)
					self._data_groups[key] = group
				elif is_removed:
					group = self._data_groups.pop(key, None)
					if group:
						try:
							self._scene.remove(group)
						except Exception:
							pass
					continue

				group = self._data_groups[key]
				group.visible = visible

				# Clear children for rebuild
				while len(group.children):
					group.remove(group.children[0])

				if not visible:
					continue

				pp = self.pparm.get(key)
				if pp is None:
					continue

				ax_cfg = self.axes.get(key)
				if ax_cfg is None:
					continue

				data_list = self.data[key]
				x_idx, y_idx, z_idx = ax_cfg
				color_hex = COLORS[pp[0] % len(COLORS)]
				do_line = pp[1]
				line_type = pp[2]
				line_width = pp[3]
				do_sym = pp[4]
				sym_type = pp[5]
				sym_size = pp[6]
				alpha = pp[10] if len(pp) > 10 else 0.8

				# Get coordinate arrays
				try:
					if x_idx == -1:
						x = np.arange(len(data_list[y_idx]), dtype=np.float32)
					elif x_idx < len(data_list):
						x = np.asarray(data_list[x_idx], dtype=np.float32)
					else:
						continue

					if y_idx < len(data_list):
						y = np.asarray(data_list[y_idx], dtype=np.float32)
					else:
						continue

					if z_idx < len(data_list):
						z = np.asarray(data_list[z_idx], dtype=np.float32)
					else:
						continue

					n = min(len(x), len(y), len(z))
					x, y, z = x[:n], y[:n], z[:n]

					if len(x) == 0:
						continue
				except Exception:
					continue

				# Transform to world coordinates (-1 to +1 cube)
				wx, wy, wz = self._data_to_world(x, y, z)

				# Mask points outside the bounding cube (don't show data past limits)
				inside = np.logical_and(
					np.logical_and(np.abs(wx) <= 1.0, np.abs(wy) <= 1.0),
					np.abs(wz) <= 1.0,
				)

				# Create line object if needed - clip segments to cube bounds
				if do_line and len(x) > 1:
					w_clipped = np.column_stack([wx, wy, wz]).clip(-1.0, 1.0).astype(np.float32)
					line_color = _hex_to_rgba(color_hex)[:3] + (alpha,)
					line_material = gfx.LineMaterial(
						thickness=line_width,
						color=line_color,
					)
					if line_type == 1:
						line_material.dash_pattern = [10, 5]
					elif line_type == 2:
						line_material.dash_pattern = [2, 3]
					elif line_type == 3:
						line_material.dash_pattern = [10, 2, 2, 2]

					line_obj = gfx.Line(gfx.Geometry(positions=w_clipped), line_material)
					group.add(line_obj)

				# Create point markers if needed (only for points inside the cube)
				if do_sym:
					marker_size = max(sym_size, 2)
					marker_nodes = self._create_markers(
						wx[inside], wy[inside], wz[inside], sym_type, marker_size, color_hex, alpha)
					for mn in marker_nodes:
						group.add(mn)

		finally:
			self._rebuilding = False

		self._redraw = True
		self._request_render()

	def _create_markers(self, wx, wy, wz, sym_type, size, color_hex, alpha):
		"""Create marker points in world coordinates."""
		n = len(wx)
		if n == 0:
			return []

		positions = np.column_stack([wx, wy, wz]).astype(np.float32)
		marker_color = _hex_to_rgba(color_hex)[:3] + (alpha,)

		shape_map = {
			0: "circle",
			1: "square",
			2: "plus",
			3: "triangle_up",
			4: "triangle_down",
		}
		marker_shape = shape_map.get(sym_type, "circle")

		material = gfx.PointsMarkerMaterial(size=size, marker=marker_shape,
											color=marker_color)
		return [gfx.Points(gfx.Geometry(positions=positions), material)]

	def set_data(self, input_data, key="data", replace=False, quiet=False,
				 color=-1, linewidth=1, linetype=-2, symtype=-2, symsize=6,
				 comments=None, column_labels=None):
		"""Set a keyed data set."""
		self._dirty = True

		if replace:
			for k in list(self.data.keys()):
				self.clear_data(k)

		if input_data is None:
			self.clear_data(key)
			return

		data_list = self._convert_data(input_data)
		if not data_list:
			return

		is_new_key = key not in self.data

		self.data[key] = data_list
		self.visibility.setdefault(key, True)

		# Determine axis mapping (x_idx, y_idx, z_idx)
		if is_new_key:
			n_cols = len(data_list)
			if n_cols <= 1:
				self.axes[key] = (-1, 0, 0)
			elif n_cols == 2:
				self.axes[key] = (0, 1, 0)
			else:
				self.axes[key] = (0, 1, 2)

		# Determine plot parameters
		if is_new_key:
			if color < 0:
				color = len(self.data) % len(COLORS)

			x_idx = self.axes[key][0]
			y_idx = self.axes[key][1]
			try:
				x_vals = data_list[x_idx] if x_idx >= 0 else np.arange(len(data_list[y_idx]))
				if len(x_vals) > 0 and (np.diff(x_vals) >= 0).all():
					do_line, lt, do_sym, st = True, 0, False, 0
				else:
					do_line, lt, do_sym, st = False, 0, True, 0
			except Exception:
				do_line, lt, do_sym, st = True, 0, False, 0

			if linetype == -2:
				lt_use = lt
			elif linetype == -1:
				do_line = False
				lt_use = 0
			else:
				do_line = True
				lt_use = linetype

			if symtype == -2:
				st_use = st
			elif symtype == -1:
				do_sym = False
				st_use = 0
			else:
				do_sym = True
				st_use = symtype
		else:
			pp = self.pparm.get(key)
			if pp:
				color = pp[0] if color < 0 else color
				lt_use = pp[2] if linetype < 0 else linetype
				linewidth = pp[3] if linewidth < 0 else linewidth
				st_use = pp[5] if symtype < 0 else symtype
				symsize = pp[6] if symsize < 0 else symsize
			else:
				lt_use = 0 if linetype < 0 else linetype
				st_use = 0 if symtype < 0 else symtype

			do_line = (linetype != -1) if linetype >= 0 else bool(pp[1] if pp else True)
			do_sym = (symtype != -1) if symtype >= 0 else bool(pp[4] if pp else False)

		self.pparm[key] = (color, do_line, lt_use, max(linewidth, 1),
						   do_sym, st_use, symsize, False, 50, 10, 0.8, 50)

		if comments is not None:
			self.comments[key] = comments

		if column_labels is not None:
			self.column_labels[key] = column_labels
			try:
				xa, ya, za = self.axes[key]
				if xa >= 0 and ya >= 0 and za >= 0:
					self.set_axis_parms(
						str(column_labels[xa]), str(column_labels[ya]),
						str(column_labels[za]))
			except Exception:
				pass

		self.autoscale()

		if self.inspector:
			self.inspector.datachange()

		if not quiet:
			self._dirty = True
			self._request_render()

	def clear_data(self, key):
		"""Remove a data set."""
		try:
			del self.data[key]
		except KeyError:
			pass
		try:
			del self.visibility[key]
		except KeyError:
			pass
		try:
			del self.axes[key]
		except KeyError:
			pass
		try:
			del self.pparm[key]
		except KeyError:
			pass
		self._dirty = True
		self._request_render()

	def _convert_data(self, input_data):
		"""Convert various input types to a list of numpy arrays."""
		if isinstance(input_data, np.ndarray) and input_data.ndim == 1:
			x_axis = np.arange(len(input_data), dtype=np.float32)
			return [x_axis, input_data.astype(np.float32)]

		if isinstance(input_data, (list, tuple)):
			result = []
			for item in input_data:
				try:
					arr = np.asarray(item, dtype=np.float32).ravel()
					result.append(arr)
				except Exception:
					return None

			return result
		return None

	def set_data_from_file(self, filename, replace=False, quiet=False):
		"""Read a keyed data set from a text file."""
		try:
			with open(filename) as f:
				rdata = f.readlines()
		except Exception as e:
			print(f"Error reading {filename}: {e}")
			return False

		if replace:
			for k in list(self.data.keys()):
				self.clear_data(k)

		rdata = [line for line in rdata if line.strip() and not line.startswith("#")]
		if not rdata:
			return False

		try:
			if "," in rdata[0]:
				rdata = [[float(j) for j in line.split(",") if j.strip()] for line in rdata]
			else:
				rdata = [[float(j) for j in line.split()] for line in rdata]

			nx = len(rdata[0])
			ny = len(rdata)
			data = [[np.array([rdata[j][i] for j in range(ny)], dtype=np.float32)] for i in range(nx)]
		except Exception as e:
			print(f"Error parsing {filename}: {e}")
			return False

		from os.path import basename
		key = basename(filename)
		self.set_data(data, key, quiet=quiet)
		return True

	def autoscale(self):
		"""Auto-scale all three axes to fit visible data."""
		all_x, all_y, all_z = [], [], []

		for key, data_list in self.data.items():
			if not self.visibility.get(key, True):
				continue
			ax_cfg = self.axes.get(key)
			if ax_cfg is None:
				continue

			x_idx, y_idx, z_idx = ax_cfg
			try:
				if x_idx == -1:
					x = np.arange(len(data_list[y_idx]), dtype=np.float32)
				elif x_idx < len(data_list):
					x = data_list[x_idx]
				else:
					continue

				y = data_list[y_idx] if y_idx < len(data_list) else None
				z = data_list[z_idx] if z_idx < len(data_list) else None

				if y is None or z is None:
					continue

				n = min(len(x), len(y), len(z))
				all_x.append(x[:n])
				all_y.append(y[:n])
				all_z.append(z[:n])
			except Exception:
				pass

		if not all_x:
			return

		x_all = np.concatenate(all_x)
		y_all = np.concatenate(all_y)
		z_all = np.concatenate(all_z)

		mask = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(z_all)
		x_finite, y_finite, z_finite = x_all[mask], y_all[mask], z_all[mask]

		if len(x_finite) == 0:
			return

		for vals, axis in [(x_finite, 'x'), (y_finite, 'y'), (z_finite, 'z')]:
			val_range = float(vals.max() - vals.min())
			padding = max(val_range * 0.05, 1e-10)
			low = float(vals.min()) - padding
			high = float(vals.max()) + padding
			if axis == 'x':
				self.xlimits = (low, high)
			elif axis == 'y':
				self.ylimits = (low, high)
			else:
				self.zlimits = (low, high)

		# Update inspector limit boxes if open
		if self.inspector:
			self.inspector._update_limits_from_widget()

	def set_axis_parms(self, xlabel="", ylabel="", zlabel=""):
		"""Set axis labels."""
		self.xaxis_label = xlabel
		self.yaxis_label = ylabel
		self.zaxis_label = zlabel

	def get_inspector(self):
		if self.inspector is None:
			from EMAN3.gui.emplot3d import EMPlot3DInspector
			self.inspector = EMPlot3DInspector(self)
		return self.inspector

	def show_inspector(self, show=True):
		if show:
			insp = self.get_inspector()
			insp.show()
			insp.raise_()
			insp.activateWindow()
			self._dirty = True

	def _on_mouse_press(self, event):
		"""Handle mouse press events."""
		pos = event.position() if hasattr(event, 'position') else event
		if event.button() == Qt.MiddleButton:
			insp = self.get_inspector()
			if insp.isVisible():
				insp.hide()
			else:
				insp.show()
				insp.raise_()
				insp.activateWindow()

		if event.button() == Qt.RightButton:
			self._right_drag_active = True
			self._right_drag_start = (pos.x(), pos.y())

	def _screen_delta_to_data(self, dx_px, dy_px):
		"""Convert screen-pixel delta to data-space offset using camera projection."""
		if self._camera is None or self.xlimits is None:
			return 0.0, 0.0

		w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
		cam_pos = np.array(self._camera.world.position)
		target = np.array(self._controller.target)
		view_dir = target - cam_pos
		dist = np.linalg.norm(view_dir)

		fov_rad = np.radians(self._camera.fov)
		vp_h = 2.0 * dist * np.tan(fov_rad / 2.0)
		aspect = w / h
		vp_w = vp_h * aspect

		px_x = vp_w / w
		py_y = vp_h / h

		# Use camera's actual screen-facing right/up vectors from its rotation matrix
		cam_rot = np.array(self._camera.world.matrix[:3, :3], dtype=np.float32)
		cam_right = cam_rot[:, 0]  # Camera local +X → screen right on any viewing angle
		cam_up = cam_rot[:, 1]    # Camera local +Y → screen up

		world_offset = -(dx_px * px_x) * cam_right + (dy_px * py_y) * cam_up

		xspan = self.xlimits[1] - self.xlimits[0]
		yspan = self.ylimits[1] - self.ylimits[0]
		zspan = self.zlimits[1] - self.zlimits[0]

		return (
			world_offset[0] * (xspan / 2.0),
			world_offset[1] * (yspan / 2.0),
			world_offset[2] * (zspan / 2.0),
		)

	def _on_mouse_move(self, event):
		if self._right_drag_active and event.modifiers() & QtCore.Qt.ShiftModifier:
			if self.xlimits is None or self.ylimits is None or self.zlimits is None:
				return
			pos = event.position() if hasattr(event, 'position') else event
			dx = pos.x() - self._right_drag_start[0]
			dy = pos.y() - self._right_drag_start[1]
			dx_data, dy_data, dz_data = self._screen_delta_to_data(dx, dy)

			xmin, xmax = self.xlimits
			ymin, ymax = self.ylimits
			zmin, zmax = self.zlimits
			self.xlimits = (xmin + dx_data, xmax + dx_data)
			self.ylimits = (ymin + dy_data, ymax + dy_data)
			self.zlimits = (zmin + dz_data, zmax + dz_data)
			if self.inspector:
				self.inspector._update_limits_from_widget()
			self._dirty = True
			self._request_render()

		if self._right_drag_active:
			pos = event.position() if hasattr(event, 'position') else event
			self._right_drag_start = (pos.x(), pos.y())

	def _on_mouse_release(self, event):
		if event.button() == Qt.RightButton:
			self._right_drag_active = False
			self._right_drag_start = None

	def _on_wheel(self, event):
		"""Shift+wheel zooms data limits; otherwise controller handles camera zoom."""
		if event.modifiers() & QtCore.Qt.ShiftModifier:
			angle = event.angleDelta()
			delta = angle.y()
			factor = 1.2 if delta < 0 else (1.0 / 1.2)
			self._zoom_data_limits(factor)
			return

	def _zoom_data_limits(self, factor):
		"""Zoom data limits uniformly on all axes by a given factor."""
		if self.xlimits is None or self.ylimits is None or self.zlimits is None:
			return

		xmin, xmax = self.xlimits
		ymin, ymax = self.ylimits
		zmin, zmax = self.zlimits

		xc = (xmin + xmax) / 2.0
		yc = (ymin + ymax) / 2.0
		zc = (zmin + zmax) / 2.0

		xs = (xmax - xmin) * factor / 2.0
		ys = (ymax - ymin) * factor / 2.0
		zs = (zmax - zmin) * factor / 2.0

		self.xlimits = (xc - xs, xc + xs)
		self.ylimits = (yc - ys, yc + ys)
		self.zlimits = (zc - zs, zc + zs)

		# Sync inspector limit boxes
		if self.inspector:
			self.inspector._update_limits_from_widget()

		self._dirty = True
		self._request_render()

	def _on_key_press(self, event):
		if event.key() == Qt.Key_C:
			self.show_inspector(True)
		elif event.key() == Qt.Key_F or event.key() == Qt.Key_R:
			self.autoscale()
			self._dirty = True
			self._request_render()
		else:
			self.keypress.emit(event)

	def closeEvent(self, event):
		if self.inspector:
			self.inspector.close()
		super().closeEvent(event)


# Color names for combo boxes
COLOR_NAMES = ["Black", "Blue", "Red", "Green", "Cyan", "Magenta", "Yellow", "Grey"]

COLOR_RGB = {
	0: QtGuiQColor(0, 0, 0),
	1: QtGuiQColor(65, 105, 225),
	2: QtGuiQColor(220, 20, 60),
	3: QtGuiQColor(34, 139, 34),
	4: QtGuiQColor(0, 206, 209),
	5: QtGuiQColor(148, 0, 211),
	6: QtGuiQColor(255, 215, 0),
	7: QtGuiQColor(128, 128, 128),
}


class EMPlot3DInspector(QtWidgets.QWidget):

	def __init__(self, target_widget):
		super().__init__()
		self.target = weakref.ref(target_widget)
		self.setWindowTitle("3D Plot Controls")
		self.resize(350, 480)

		self._selection_suppressed = False

		self._build_ui()
		self._wire_signals()
		self._sync_from_widget()
		self.xmin_box.valueChanged.connect(self._on_limit_change)
		self.xmax_box.valueChanged.connect(self._on_limit_change)
		self.ymin_box.valueChanged.connect(self._on_limit_change)
		self.ymax_box.valueChanged.connect(self._on_limit_change)
		self.zmin_box.valueChanged.connect(self._on_limit_change)
		self.zmax_box.valueChanged.connect(self._on_limit_change)

	def _get_tgt(self):
		return self.target() if self.target else None

	def _build_ui(self):
		vbl = QtWidgets.QVBoxLayout(self)
		vbl.setContentsMargins(6, 6, 6, 6)
		vbl.setSpacing(4)

		# ── Data set list ──
		dsg = QtWidgets.QGroupBox("Data sets")
		dsl = QtWidgets.QVBoxLayout(dsg)

		self.setlist = QtWidgets.QListWidget()
		self.setlist.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
		self.setlist.setToolTip("Select data sets to view and edit. Check/uncheck to toggle visibility.")
		dsl.addWidget(self.setlist)

		h_sel = QtWidgets.QHBoxLayout()
		self.none_but = QtWidgets.QPushButton("None")
		self.none_but.clicked.connect(self.sel_none)
		self.none_but.setToolTip("Deselect all data sets")
		h_sel.addWidget(self.none_but)
		self.all_but = QtWidgets.QPushButton("All")
		self.all_but.clicked.connect(self.sel_all)
		self.all_but.setToolTip("Select all data sets")
		h_sel.addWidget(self.all_but)
		dsl.addLayout(h_sel)

		self.showslide = ValSlider(label="Sel:", value=0)
		self.showslide.setIntonly(True)
		self.showslide.setRange(0, 30)
		self.showslide.setToolTip("Show only the data set at this index in the list")
		dsl.addWidget(self.showslide)

		h_slide_vals = QtWidgets.QHBoxLayout()
		self.nbox = ValBox(label="ns:", value=1)
		self.nbox.setIntonly(True)
		self.nbox.setToolTip("Number of consecutive sets to show with slider")
		h_slide_vals.addWidget(self.nbox)
		self.stepbox = ValBox(label="stp:", value=1)
		self.stepbox.setIntonly(True)
		self.stepbox.setToolTip("Step size between shown sets with slider")
		h_slide_vals.addWidget(self.stepbox)
		dsl.addLayout(h_slide_vals)

		vbl.addWidget(dsg)

		# ── Buttons ──
		btn_row = QtWidgets.QHBoxLayout()
		self.stats_btn = QtWidgets.QPushButton("Statistics")
		self.stats_btn.clicked.connect(self._on_statistics)
		self.stats_btn.setToolTip("Print min/max/mean/std for each column of selected data")
		btn_row.addWidget(self.stats_btn)
		self.regress_btn = QtWidgets.QPushButton("Regression")
		self.regress_btn.clicked.connect(self._on_regression)
		self.regress_btn.setToolTip("Fit linear regression to selected columns")
		btn_row.addWidget(self.regress_btn)
		vbl.addLayout(btn_row)

		# ── Appearance controls ──
		apg = QtWidgets.QGroupBox("Appearance")
		apl = QtWidgets.QVBoxLayout(apg)

		self.color_combo = QtWidgets.QComboBox()
		self.color_combo.addItems(COLOR_NAMES)
		self.color_combo.setToolTip("Set color for selected data sets")
		h_color = QtWidgets.QHBoxLayout()
		h_color.addWidget(QtWidgets.QLabel("Color:"))
		h_color.addWidget(self.color_combo)
		apl.addLayout(h_color)

		# 2x3 grid: Line/Symbol toggles + params
		ag = QtWidgets.QGridLayout()

		self.line_tog = QtWidgets.QPushButton("Line")
		self.line_tog.setCheckable(True)
		self.line_tog.clicked.connect(self._on_appearance_change)
		self.line_tog.setToolTip("Toggle line display for selected data sets")
		self.sym_tog = QtWidgets.QPushButton("Symbol")
		self.sym_tog.setCheckable(True)
		self.sym_tog.clicked.connect(self._on_appearance_change)
		self.sym_tog.setToolTip("Toggle symbol markers for selected data sets")
		ag.addWidget(self.line_tog, 0, 0)
		ag.addWidget(self.sym_tog, 0, 1)

		self.linetype_combo = QtWidgets.QComboBox()
		self.linetype_combo.addItems(["Solid", "Dashed", "Dotted", "Dash-Dot"])
		self.linetype_combo.setToolTip("Set line style for selected data sets")
		self.linewidth_spin = QtWidgets.QSpinBox()
		self.linewidth_spin.setRange(1, 10)
		self.linewidth_spin.valueChanged.connect(self._on_appearance_change)
		self.linewidth_spin.setToolTip("Set line thickness (1-10 pixels)")
		ag.addWidget(self.linetype_combo, 1, 0)
		ag.addWidget(self.linewidth_spin, 1, 1)

		self.symtype_combo = QtWidgets.QComboBox()
		self.symtype_combo.addItems(["Circle", "Square", "Plus", "TriUp", "TriDown"])
		self.symtype_combo.setToolTip("Set marker shape for selected data sets")
		self.symsize_spin = QtWidgets.QSpinBox()
		self.symsize_spin.setRange(1, 30)
		self.symsize_spin.valueChanged.connect(self._on_appearance_change)
		self.symsize_spin.setToolTip("Set marker size (1-30 pixels)")
		ag.addWidget(self.symtype_combo, 2, 0)
		ag.addWidget(self.symsize_spin, 2, 1)

		apl.addLayout(ag)
		vbl.addWidget(apg)

		# ── Columns + Limits merged grid: Col | Min | Max per axis row ──
		col_limit_g = QtWidgets.QGridLayout()

		# X axis row: Col selector, min box, max box
		xlabel = QtWidgets.QLabel("X col:")
		xlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
		col_limit_g.addWidget(xlabel, 0, 0)
		self.col_x_spin = QtWidgets.QSpinBox()
		self.col_x_spin.setRange(-1, 20)
		self.col_x_spin.setToolTip("Column index for X axis (-1 = row index)")
		col_limit_g.addWidget(self.col_x_spin, 0, 1)
		self.xmin_box = ValBox(label="min:", value=0)
		self.xmin_box.setToolTip("X axis minimum")
		col_limit_g.addWidget(self.xmin_box, 0, 2)
		self.xmax_box = ValBox(label="max:", value=1)
		self.xmax_box.setToolTip("X axis maximum")
		col_limit_g.addWidget(self.xmax_box, 0, 3)

		# Y axis row
		ylabel = QtWidgets.QLabel("Y col:")
		ylabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
		col_limit_g.addWidget(ylabel, 1, 0)
		self.col_y_spin = QtWidgets.QSpinBox()
		self.col_y_spin.setRange(-1, 20)
		self.col_y_spin.setToolTip("Column index for Y axis")
		col_limit_g.addWidget(self.col_y_spin, 1, 1)
		self.ymin_box = ValBox(label="min:", value=0)
		self.ymin_box.setToolTip("Y axis minimum")
		col_limit_g.addWidget(self.ymin_box, 1, 2)
		self.ymax_box = ValBox(label="max:", value=1)
		self.ymax_box.setToolTip("Y axis maximum")
		col_limit_g.addWidget(self.ymax_box, 1, 3)

		# Z axis row
		zlabel = QtWidgets.QLabel("Z col:")
		zlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
		col_limit_g.addWidget(zlabel, 2, 0)
		self.col_z_spin = QtWidgets.QSpinBox()
		self.col_z_spin.setRange(0, 20)
		self.col_z_spin.setToolTip("Column index for Z axis")
		col_limit_g.addWidget(self.col_z_spin, 2, 1)
		self.zmin_box = ValBox(label="min:", value=0)
		self.zmin_box.setToolTip("Z axis minimum")
		col_limit_g.addWidget(self.zmin_box, 2, 2)
		self.zmax_box = ValBox(label="max:", value=1)
		self.zmax_box.setToolTip("Z axis maximum")
		col_limit_g.addWidget(self.zmax_box, 2, 3)

		vbl.addLayout(col_limit_g)

		# ── Axis labels ──
		label_g = QtWidgets.QGridLayout()
		self.xlabel_edit = QtWidgets.QLineEdit()
		self.xlabel_edit.setToolTip("X axis title text")
		self.ylabel_edit = QtWidgets.QLineEdit()
		self.ylabel_edit.setToolTip("Y axis title text")
		self.zlabel_edit = QtWidgets.QLineEdit()
		self.zlabel_edit.setToolTip("Z axis title text")
		label_g.addWidget(QtWidgets.QLabel("X:"), 0, 0)
		label_g.addWidget(self.xlabel_edit, 0, 1)
		label_g.addWidget(QtWidgets.QLabel("Y:"), 1, 0)
		label_g.addWidget(self.ylabel_edit, 1, 1)
		label_g.addWidget(QtWidgets.QLabel("Z:"), 2, 0)
		label_g.addWidget(self.zlabel_edit, 2, 1)
		vbl.addLayout(label_g)

		# ── Alpha slider ──
		self.alpha_slider = ValSlider(label="Alpha", value=0.8)
		self.alpha_slider.setRange(0.1, 1.0)
		self.alpha_slider.setToolTip("Transparency for selected data sets")
		vbl.addWidget(self.alpha_slider)

		# ── Rescale button ──
		self.rescale_btn = QtWidgets.QPushButton("Rescale")
		self.rescale_btn.clicked.connect(self._on_rescale)
		self.rescale_btn.setToolTip("Auto-scale all axes to fit visible data")
		vbl.addWidget(self.rescale_btn)

		vbl.addStretch()

	def _wire_signals(self):
		tgt = self._get_tgt()
		if not tgt:
			return

		self.setlist.currentRowChanged.connect(self._on_selection_changed)
		self.color_combo.currentIndexChanged.connect(self._on_color_change)
		self.linetype_combo.currentIndexChanged.connect(self._on_appearance_change)
		self.linewidth_spin.valueChanged.connect(self._on_appearance_change)
		self.symtype_combo.currentIndexChanged.connect(self._on_appearance_change)
		self.symsize_spin.valueChanged.connect(self._on_appearance_change)
		self.col_x_spin.valueChanged.connect(self._on_column_change)
		self.col_y_spin.valueChanged.connect(self._on_column_change)
		self.col_z_spin.valueChanged.connect(self._on_column_change)

		self.xlabel_edit.editingFinished.connect(self._on_label_change)
		self.ylabel_edit.editingFinished.connect(self._on_label_change)
		self.zlabel_edit.editingFinished.connect(self._on_label_change)

		self.alpha_slider.valueChanged.connect(self._on_alpha_change)
		self.setlist.itemChanged.connect(self._on_item_changed)
		self.showslide.valueChanged.connect(self._on_slide_change)

	def _sync_from_widget(self):
		tgt = self._get_tgt()
		if not tgt:
			return

		if tgt.xlimits is not None:
			self.xmin_box.setValue(tgt.xlimits[0])
			self.xmax_box.setValue(tgt.xlimits[1])
		if tgt.ylimits is not None:
			self.ymin_box.setValue(tgt.ylimits[0])
			self.ymax_box.setValue(tgt.ylimits[1])
		if tgt.zlimits is not None:
			self.zmin_box.setValue(tgt.zlimits[0])
			self.zmax_box.setValue(tgt.zlimits[1])

		self.xlabel_edit.setText(tgt.xaxis_label)
		self.ylabel_edit.setText(tgt.yaxis_label)
		self.zlabel_edit.setText(tgt.zaxis_label)

		self.update_list()

	def _update_limits_from_widget(self):
		tgt = self._get_tgt()
		if not tgt:
			return

		try:
			self.xmin_box.valueChanged.disconnect(self._on_limit_change)
			self.xmax_box.valueChanged.disconnect(self._on_limit_change)
			self.ymin_box.valueChanged.disconnect(self._on_limit_change)
			self.ymax_box.valueChanged.disconnect(self._on_limit_change)
			self.zmin_box.valueChanged.disconnect(self._on_limit_change)
			self.zmax_box.valueChanged.disconnect(self._on_limit_change)
		except Exception:
			pass

		if tgt.xlimits is not None:
			self.xmin_box.setValue(tgt.xlimits[0])
			self.xmax_box.setValue(tgt.xlimits[1])
		if tgt.ylimits is not None:
			self.ymin_box.setValue(tgt.ylimits[0])
			self.ymax_box.setValue(tgt.ylimits[1])
		if tgt.zlimits is not None:
			self.zmin_box.setValue(tgt.zlimits[0])
			self.zmax_box.setValue(tgt.zlimits[1])

		self.xmin_box.valueChanged.connect(self._on_limit_change)
		self.xmax_box.valueChanged.connect(self._on_limit_change)
		self.ymin_box.valueChanged.connect(self._on_limit_change)
		self.ymax_box.valueChanged.connect(self._on_limit_change)
		self.zmin_box.valueChanged.connect(self._on_limit_change)
		self.zmax_box.valueChanged.connect(self._on_limit_change)

	def _update_controls_for_key(self, key):
		tgt = self._get_tgt()
		if not tgt or key is None:
			return

		pp = tgt.pparm.get(key)
		if pp is None:
			return

		color_idx = pp[0] % len(COLORS)
		self.color_combo.setCurrentIndex(color_idx)

		self.line_tog.setChecked(bool(pp[1]))
		lt = min(pp[2], 3) if pp[2] >= 0 else 0
		self.linetype_combo.setCurrentIndex(lt)
		self.linewidth_spin.setValue(max(1, pp[3]))

		self.sym_tog.setChecked(bool(pp[4]))
		st = min(pp[5], 4) if pp[5] >= 0 else 0
		self.symtype_combo.setCurrentIndex(st)
		self.symsize_spin.setValue(max(1, pp[6]))

		alpha = pp[10] if len(pp) > 10 else 0.8
		self.alpha_slider.setValue(alpha)

		try:
			self.col_x_spin.valueChanged.disconnect(self._on_column_change)
			self.col_y_spin.valueChanged.disconnect(self._on_column_change)
			self.col_z_spin.valueChanged.disconnect(self._on_column_change)
		except Exception:
			pass

		max_cols = 0
		for k, dl in tgt.data.items():
			if isinstance(dl, list) and len(dl) > max_cols:
				max_cols = len(dl)
		self.col_x_spin.setRange(-1, max(5, max_cols - 1))
		self.col_y_spin.setRange(-1, max(5, max_cols - 1))
		self.col_z_spin.setRange(0, max(5, max_cols - 1))

		ax_cfg = tgt.axes.get(key, (-1, 0, 0))
		self.col_x_spin.setValue(ax_cfg[0])
		self.col_y_spin.setValue(ax_cfg[1])
		self.col_z_spin.setValue(ax_cfg[2])

		self.col_x_spin.valueChanged.connect(self._on_column_change)
		self.col_y_spin.valueChanged.connect(self._on_column_change)
		self.col_z_spin.valueChanged.connect(self._on_column_change)

	def _on_selection_changed(self, row):
		if self._selection_suppressed:
			return
		item = self.setlist.selectedItems()[0] if self.setlist.selectedItems() else None
		if item is None:
			return
		key = item.data(QtCore.Qt.UserRole)
		self._update_controls_for_key(key)

	def _selected_keys(self):
		keys = []
		for item in self.setlist.selectedItems():
			key = item.data(QtCore.Qt.UserRole)
			if key is not None:
				keys.append(key)
		return keys

	def _selected_key(self):
		keys = self._selected_keys()
		return keys[0] if keys else None

	def sel_all(self):
		tgt = self._get_tgt()
		if tgt:
			for key in tgt.visibility:
				tgt.visibility[key] = True
			self.update_list()
			tgt._dirty = True
			tgt._request_render()

	def sel_none(self):
		tgt = self._get_tgt()
		if tgt:
			for key in tgt.visibility:
				tgt.visibility[key] = False
			self.update_list()
			tgt._dirty = True
			tgt._request_render()

	def _on_item_changed(self, item):
		key = item.data(QtCore.Qt.UserRole)
		if key is None:
			return
		tgt = self._get_tgt()
		if not tgt:
			return
		checked = (item.checkState() == QtCore.Qt.Checked)
		tgt.visibility[key] = checked
		tgt._dirty = True
		tgt._request_render()

	def _on_slide_change(self, val):
		tgt = self._get_tgt()
		if not tgt:
			return
		rng_n0 = int(val)
		rng_n1 = int(self.nbox.getValue()) if self.nbox else 1
		rng_stp = int(self.stepbox.getValue()) if self.stepbox else 1
		range_list = list(range(rng_n0, rng_n0 + rng_stp * rng_n1, rng_stp))
		sorted_keys = sorted(tgt.visibility.keys(), key=lambda k: str(k))
		for i, k in enumerate(sorted_keys):
			tgt.visibility[k] = (i in range_list)
		self.update_list()
		tgt._dirty = True
		tgt._request_render()

	def update_list(self):
		tgt = self._get_tgt()
		if not tgt:
			return

		old_row = self.setlist.currentRow()

		self.setlist.blockSignals(True)
		try:
			self.setlist.clear()

			for key in tgt.data:
				pp = tgt.pparm.get(key)
				color_idx = pp[0] % len(COLORS) if pp else 0

				item = QtWidgets.QListWidgetItem(key)
				item.setData(QtCore.Qt.UserRole, key)

				flags = item.flags()
				flags |= QtCore.Qt.ItemIsUserCheckable
				item.setFlags(flags)

				if tgt.visibility.get(key, True):
					item.setCheckState(QtCore.Qt.Checked)
				else:
					item.setCheckState(QtCore.Qt.Unchecked)

				if color_idx in COLOR_RGB:
					item.setForeground(COLOR_RGB[color_idx])

				self.setlist.addItem(item)
		finally:
			self.setlist.blockSignals(False)

		num_sets = self.setlist.count()
		try:
			self.showslide.valueChanged.disconnect(self._on_slide_change)
		except Exception:
			pass
		if num_sets > 1:
			self.showslide.setRange(0, num_sets - 1)
		else:
			self.showslide.setRange(0, 0)
		self.showslide.valueChanged.connect(self._on_slide_change)

		if 0 <= old_row < self.setlist.count():
			self._selection_suppressed = True
			try:
				self.setlist.setCurrentRow(old_row)
			finally:
				self._selection_suppressed = False

		if old_row < 0 and self.setlist.count() > 0:
			self._selection_suppressed = True
			try:
				self.setlist.setCurrentRow(0)
			finally:
				self._selection_suppressed = False
			item = self.setlist.item(0)
			if item is not None:
				key = item.data(QtCore.Qt.UserRole)
				self._update_controls_for_key(key)

	def datachange(self):
		self.update_list()

	def _on_statistics(self):
		key = self._selected_key()
		if key is None or not self.target():
			return
		tgt = self.target()
		data_list = tgt.data.get(key)
		if not data_list:
			return

		n_cols = len(data_list)
		max_len = max(len(col) for col in data_list) if data_list else 0

		header = "Column\tMin\tMax\tMean\tStd"
		print(header)
		print("-" * len(header))

		for j in range(n_cols):
			col = np.asarray(data_list[j], dtype=np.float64)[:max_len]
			valid = col[np.isfinite(col)]
			if len(valid) == 0:
				print(f"{j}\t---\t---\t---\t---")
				continue
			mn, mx = float(valid.min()), float(valid.max())
			mean = float(np.mean(valid))
			std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
			print(f"{j}\t{mn:.6g}\t{mx:.6g}\t{mean:.6g}\t{std:.6g}")

	def _on_regression(self):
		key = self._selected_key()
		if key is None or not self.target():
			return
		tgt = self.target()
		ax_cfg = tgt.axes.get(key)
		if ax_cfg is None:
			return

		x_idx, y_idx, z_idx = ax_cfg
		data_list = tgt.data[key]

		try:
			x_vals = np.asarray(data_list[x_idx], dtype=np.float64) if x_idx >= 0 else np.arange(len(data_list[y_idx]))
			y_vals = np.asarray(data_list[y_idx], dtype=np.float64)

			mask = np.isfinite(x_vals) & np.isfinite(y_vals)
			xd, yd = x_vals[mask], y_vals[mask]

			if len(xd) < 2:
				return

			A = np.vstack([xd, np.ones(len(xd))]).T
			m, b = np.linalg.lstsq(A, yd, rcond=None)[0]

			new_key = f"reg_{key}"
			n_cols = len(data_list)
			# Compute range with 5% padding
			x_range = float(xd.max() - xd.min()) if xd.max() != xd.min() else 1.0
			padding = x_range * 0.05

			x_fit = np.linspace(float(xd.min()) - padding, float(xd.max()) + padding, 20)
			y_fit = m * x_fit + b

			new_cols = [np.zeros_like(x_fit)] * n_cols
			for j in range(n_cols):
				new_cols[j] = np.zeros(len(x_fit))
			if x_idx < n_cols:
				new_cols[x_idx] = x_fit.astype(np.float32)
			if y_idx < n_cols:
				new_cols[y_idx] = y_fit.astype(np.float32)

			tgt.set_data(new_cols, key=new_key)

			print(f"Regression: y = {m:.8g} * x + {b:.8g}")
		except Exception as e:
			print(f"Regression error: {e}")

	def _on_appearance_change(self):
		keys = self._selected_keys()
		if not keys or not self.target():
			return
		tgt = self.target()
		sender = self.sender()

		for key in keys:
			pp = tgt.pparm.get(key)
			if pp is None:
				continue
			pp = list(pp)
			if sender == self.line_tog:
				pp[1] = self.line_tog.isChecked()
			elif sender == self.linetype_combo:
				pp[2] = self.linetype_combo.currentIndex()
			elif sender == self.linewidth_spin:
				pp[3] = max(1, self.linewidth_spin.value())
			elif sender == self.sym_tog:
				pp[4] = self.sym_tog.isChecked()
			elif sender == self.symtype_combo:
				pp[5] = self.symtype_combo.currentIndex()
			elif sender == self.symsize_spin:
				pp[6] = max(1, self.symsize_spin.value())
			else:
				continue
			tgt.pparm[key] = pp

		tgt._dirty = True
		tgt._request_render()

	def _on_color_change(self):
		keys = self._selected_keys()
		if not keys or not self.target():
			return
		tgt = self.target()
		color_idx = self.color_combo.currentIndex()
		for key in keys:
			pp = tgt.pparm.get(key)
			if pp is None:
				continue
			pp = list(pp)
			pp[0] = color_idx
			tgt.pparm[key] = pp
		tgt._dirty = True
		tgt._request_render()

	def _on_column_change(self):
		key = self._selected_key()
		if key is None or not self.target():
			return
		tgt = self.target()
		x_idx = self.col_x_spin.value()
		y_idx = self.col_y_spin.value()
		z_idx = self.col_z_spin.value()
		tgt.axes[key] = (x_idx, y_idx, z_idx)

		tgt.autoscale()
		tgt._dirty = True
		tgt._request_render()

	def _on_rescale(self):
		tgt = self._get_tgt()
		if tgt:
			tgt.autoscale()
			tgt._dirty = True
			tgt._request_render()
			if tgt.xlimits is not None:
				self.xmin_box.setValue(tgt.xlimits[0])
				self.xmax_box.setValue(tgt.xlimits[1])
			if tgt.ylimits is not None:
				self.ymin_box.setValue(tgt.ylimits[0])
				self.ymax_box.setValue(tgt.ylimits[1])
			if tgt.zlimits is not None:
				self.zmin_box.setValue(tgt.zlimits[0])
				self.zmax_box.setValue(tgt.zlimits[1])

	def _on_limit_change(self):
		tgt = self._get_tgt()
		if not tgt:
			return
		try:
			xmin = float(self.xmin_box.getValue())
			xmax = float(self.xmax_box.getValue())
			ymin = float(self.ymin_box.getValue())
			ymax = float(self.ymax_box.getValue())
			zmin = float(self.zmin_box.getValue())
			zmax = float(self.zmax_box.getValue())

			tgt.xlimits = (xmin, xmax)
			tgt.ylimits = (ymin, ymax)
			tgt.zlimits = (zmin, zmax)

			tgt._dirty = True
			tgt._request_render()
		except Exception:
			pass

	def _on_label_change(self):
		tgt = self._get_tgt()
		if not tgt:
			return
		tgt.xaxis_label = self.xlabel_edit.text()
		tgt.yaxis_label = self.ylabel_edit.text()
		tgt.zaxis_label = self.zlabel_edit.text()
		tgt._create_axis_labels()
		tgt._dirty = True
		tgt._request_render()

	def _on_alpha_change(self, val):
		keys = self._selected_keys()
		if not keys or not self.target():
			return
		tgt = self.target()
		for key in keys:
			pp = tgt.pparm.get(key)
			if pp is None:
				continue
			pp = list(pp)
			if len(pp) > 10:
				pp[10] = val
			else:
				pp.append(val)
			tgt.pparm[key] = pp
		tgt._dirty = True
		tgt._request_render()

	def closeEvent(self, event):
		super().closeEvent(event)


def main():
	"""Test program for 3D plot widget."""
	import sys
	import numpy as np

	from PySide6.QtWidgets import QApplication

	app = QApplication(sys.argv)
	window = EMPlot3DWidget(app)

	if len(sys.argv) == 1:
		n = 200
		t = np.linspace(0, 4 * np.pi, n)
		x = np.cos(t)
		y = np.sin(t)
		z = np.linspace(-1, 1, n)
		window.set_data([x, y, z], "spiral", symtype=0, symsize=3)

		ns = 80
		phi = np.linspace(0, np.pi, ns)
		theta = np.linspace(0, 2 * np.pi, ns)
		PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
		sx = np.sin(PHI) * np.cos(THETA)
		sy = np.sin(PHI) * np.sin(THETA)
		sz = np.cos(PHI)
		window.set_data(
			[sx.ravel(), sy.ravel(), sz.ravel()],
			"sphere",
			symtype=0,
			symsize=2,
			color=1,
		)
	else:
		for i in range(1, len(sys.argv)):
			window.set_data_from_file(sys.argv[i])

	window.show()
	window.setWindowTitle("EMAN3 3D Plot Test")

	if len(sys.argv) == 1:
		window.autoscale()

	sys.exit(app.exec())


if __name__ == "__main__":
	main()
