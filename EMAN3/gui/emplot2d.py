#!/usr/bin/env python
#
# EMAN3 2D Plot Widget - PyGfx rendering with PySide6
# Ported from EMAN2 qtgui/emplot2d.py

import weakref
from math import sqrt

import pygfx as gfx
from rendercanvas.qt import QRenderWidget

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor as QtGuiQColor
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

LINE_TYPES = ["Solid", "Dashed", "Dotted", "Dash-Dot"]


def _hex_to_rgba(hex_color):
	"""Convert hex color to (r,g,b,a) tuple of floats."""
	hex_color = hex_color.lstrip("#")
	r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
	return (r / 255.0, g / 255.0, b / 255.0, 1.0)


class _EMCanvas(QRenderWidget):
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


class EMPlot2DWidget(QtWidgets.QWidget):
	"""2D plot widget using PyGfx for rendering."""

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
		self.axes = {}
		self.pparm = {}
		self.comments = {}
		self.column_labels = {}

		# Display params
		self.xlimits = None
		self.ylimits = None
		self.climits = None
		self.slimits = None
		self.plottitle = ""
		self.xaxis_label = ""
		self.yaxis_label = ""
		self.xlog = False
		self.ylog = False

		# Selection
		self.selectpoints = True
		self.selected = []
		self.mouseemit = False

		# Mouse state
		self._right_drag_pos = None
		self._left_drag_active = False
		self.shapes = {}  # shape data dict for crosshairs/zoom-box/labels
		self._shape_group = None
		self._prev_shapes = {}  # track prev shapes to avoid redundant rebuilds
		self._prev_xlabel = ""  # cache axis labels to detect changes
		self._prev_ylabel = ""

		# PyGfx setup
		self._canvas = None
		self._renderer = None
		self._scene = gfx.Scene()
		self._camera = None
		self._controller = None
		self._grid = None
		self._ruler_x = None
		self._ruler_y = None
		self._axis_labels = []
		self._data_nodes = {}
		self._border_box = None
		self._dirty = True
		self._rebuilding = False
		self._pending_rebuild = False
		self._data_groups = {}

		self._setup_gfx()

		# Inspector
		self.inspector = None

		self.resize(640, 480)

		# Defer scene init to after first paint (renderer needs valid size)
		QtCore.QTimer.singleShot(0, self._init_scene)

	def _setup_gfx(self):
		"""Initialize PyGfx rendering pipeline."""
		self._canvas = _EMCanvas(parent_widget=self, parent=self)

		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self._canvas)

		self._renderer = gfx.WgpuRenderer(self._canvas)

		# White background for publication style
		self._scene.add(gfx.Background.from_color("#ffffff"))

		# Grid for plot background - disabled for publication style
		# grid_material = gfx.GridMaterial(
		# 	major_step=1,
		# 	minor_step=0,
		# 	thickness_space="screen",
		# 	major_thickness=1,
		# 	minor_thickness=0,
		# 	infinite=True,
		# )
		# self._grid = gfx.Grid(None, grid_material, orientation="xy")
		# self._grid.visible = False

		# Rulers for axis tick labels - all 4 edges. The directions here make no visual sense, but these are correct
		self._ruler_x_bottom = gfx.Ruler(tick_side="right", tick_marker="tick_left",
								 min_tick_distance=40, color=(0, 0, 0, 1.0))
		self._ruler_y_left = gfx.Ruler(tick_side="left", tick_marker="tick_right",
							   min_tick_distance=35, color=(0, 0, 0, 1.0))
		self._ruler_x_top = gfx.Ruler(tick_side="left", tick_marker="tick_right",
							    min_tick_distance=40, color=(0, 0, 0, 1.0))
		self._ruler_y_right = gfx.Ruler(tick_side="right", tick_marker="tick_left",
								min_tick_distance=35, color=(0, 0, 0, 1.0))

		# Hide numeric tick labels on top and right rulers (tick marks only)
		self._ruler_x_top.text.material.opacity = 0
		self._ruler_y_right.text.material.opacity = 0

		# Make label text black for visible-label rulers
		self._ruler_x_bottom.text.material.color = (0, 0, 0, 1.0)
		self._ruler_y_left.text.material.color = (0, 0, 0, 1.0)
		self._camera = gfx.OrthographicCamera(maintain_aspect=False)
		# Offscreen rendering + blitting flips Y, so pre-compensate with scale_y=-1
		# to make positive data Y appear at the top of the screen
		self._camera.local.scale_y = -1

		# Ensure line materials are initialized early by adding an invisible placeholder
		_dummy_verts = np.array([[0, 0, 2], [0, 0, 2]], dtype=np.float32)
		self._line_init_placeholder = gfx.Line(
			gfx.Geometry(positions=_dummy_verts),
			gfx.LineMaterial(thickness=1, color=(0, 0, 0, 0)))
		self._scene.add(self._line_init_placeholder)

		# Ensure PointsMarkerMaterial is initialized early
		_dummy_pts = np.array([[0, 0, 2]], dtype=np.float32)
		self._point_init_placeholder = gfx.Points(
			gfx.Geometry(positions=_dummy_pts),
			gfx.PointsMarkerMaterial(size=1, marker='circle', color=(0, 0, 0, 0)))
		self._scene.add(self._point_init_placeholder)

		# Ensure Text/font renderer is primed early so labels appear immediately
		try:
			_dummy_text = gfx.Text("Init", font_size=12, render_order=999)
			_dummy_text.material.color = (0, 0, 0, 0)
			_dummy_text.local.position = (0, 0, 2)
			self._scene.add(_dummy_text)
		except Exception:
			pass

		self._controller = None

		# self._scene.add(self._grid, self._ruler_x_bottom, self._ruler_y_left,
		# 				self._ruler_x_top, self._ruler_y_right)
		self._scene.add(self._ruler_x_bottom, self._ruler_y_left,
						self._ruler_x_top, self._ruler_y_right)

		# Render loop
		self._canvas.request_draw(self._render_callback)

	def _screen_to_world(self, sx, sy):
		"""Convert screen pixel coordinates to plot data coordinates."""
		w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
		try:
			f = self._camera.frustum
			near = f[0]
			xmin, xmax = float(near[:, 0].min()), float(near[:, 0].max())
			ymin, ymax = float(near[:, 1].min()), float(near[:, 1].max())
			# X: normal mapping
			wx = xmin + (sx / w) * (xmax - xmin)
			# Y: screen sy=0 → world ymax (top of view), sy=h → ymin (bottom)
			wy = ymax - (sy / h) * (ymax - ymin)
			return wx, wy
		except Exception:
			nx = sx / w * 2 - 1
			y = 1.0 - (sy / h) * 2  # screen top(0)→NDC y=+1, bottom(h)→y=-1
			world = la.vec_unproject((nx, y), self._camera.camera_matrix, depth=0.5)
			return (float(world[0]), float(world[1]))

	def _update_camera_limits(self):
		"""Update camera to match xlimits/ylimits if set."""
		if self.xlimits is not None and self.ylimits is not None:
			xmin, xmax = self.xlimits
			ymin, ymax = self.ylimits
			# Validate that limits span a real area before updating camera
			if xmin < xmax and ymin < ymax:
				x_span = xmax - xmin
				y_span = ymax - ymin
				# Clamp extreme aspect ratios to prevent show_rect from failing
				aspect_ratio = x_span / y_span
				if aspect_ratio > 50 or aspect_ratio < 0.02:
					if aspect_ratio > 50:
						y_center = (ymin + ymax) / 2
						y_span_new = x_span / 25
						ymin, ymax = y_center - y_span_new/2, y_center + y_span_new/2
					else:
						x_center = (xmin + xmax) / 2
						x_span_new = y_span * 25
						xmin, xmax = x_center - x_span_new/2, x_center + x_span_new/2
			try:
				self._camera.show_rect(xmin, xmax, ymax, ymin)
			except Exception:
				# show_rect failed - sync limits back to what camera actually shows
				try:
					f = self._camera.frustum
					near = f[0]
					self.xlimits = (float(near[:, 0].min()), float(near[:, 0].max()))
					self.ylimits = (float(near[:, 1].min()), float(near[:, 1].max()))
				except Exception:
					pass

	def _request_render(self):
		"""Request a render frame — only fires when something actually needs redrawing."""
		if self._dirty or self.shapes:
			self._canvas.request_draw(self._render_callback)

	def _render_callback(self):
		"""Render frame — called only when something changed."""
		try:
			# Defer scene graph modifications to outside the render callback
			if self._dirty and not self._rebuilding and not self._pending_rebuild:
				self._rebuilding = True
				self._dirty = False
				# Schedule rebuild in next event-loop tick (after rendering)
				self._pending_rebuild = True
				QtCore.QTimer.singleShot(0, self._rebuild_data_groups)

			self._render_plot()

			# Only schedule another frame if there's still pending work
			if self._dirty or self.shapes:
				self._canvas.request_draw(self._render_callback)
		except Exception as e:
			print(f"Plot render error: {e}")
			import traceback
			traceback.print_exc()

	def _init_scene(self):
		"""One-time scene graph initialization (called after first paint)."""
		try:
			w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
			if w == 0:
				w, h = 640, 480

			margin_left = 65
			margin_bottom = 50
			margin_top = 20
			margin_right = 20

			# Data groups - one per key (initially empty)
			for key in self.data:
				group = gfx.Group()
				group.name = f"data-{key}"
				self._scene.add(group)
				self._data_groups[key] = group

			# Create axis labels once (screen_space)
			label_color = (0, 0, 0, 1.0)

			if self.xaxis_label:
				xlbl = gfx.Text(self.xaxis_label, font_size=18,
									screen_space=True)
				xlbl.material.color = label_color
				sx_lbl = margin_left + (w - margin_left - margin_right) / 2
				sy_lbl = h - margin_bottom + 25
				xlbl._screen_pos = (sx_lbl, sy_lbl)
				xlbl.local.position = (*self._screen_to_world(sx_lbl, sy_lbl), 0)
				self._scene.add(xlbl)
				self._axis_labels.append(xlbl)

			if self.yaxis_label:
				ylbl = gfx.Text(self.yaxis_label, font_size=18,
									screen_space=True)
				ylbl.material.color = label_color
				sy_lbl = margin_top + (h - margin_top - margin_bottom) / 2 - 10
				ylbl._screen_pos = (10, sy_lbl)
				ylbl.local.position = (*self._screen_to_world(10, sy_lbl), 0)
				ylbl.local.rotation = la.quat_from_euler((0, 0, np.radians(90)))  # Rotate 90° CCW
				self._scene.add(ylbl)
				self._axis_labels.append(ylbl)

			# Cache initial label values for change detection
			self._prev_xlabel = self.xaxis_label
			self._prev_ylabel = self.yaxis_label

			# Border box outline
			box_pts = np.zeros((5, 3), dtype=np.float32)
			self._border_box = gfx.Line(
				gfx.Geometry(positions=box_pts),
				gfx.LineMaterial(thickness=1.5, color=(0, 0, 0, 1.0)))
			self._scene.add(self._border_box)

			# Populate data groups with initial content
			self._rebuild_data_groups()
		except Exception as e:
			print(f"Scene init error: {e}")
			import traceback
			traceback.print_exc()

	def _rebuild_data_groups(self):
		"""Rebuild data groups - called outside render callback (timer or direct)."""
		if self._pending_rebuild:
			self._pending_rebuild = False
		self._rebuilding = True
		try:
			# Process all data sets - rebuild only changed or new keys
			keys = list(self.data.keys())
			old_keys = list(self._data_groups.keys())
			all_keys = sorted(set(keys + old_keys))

			for key in all_keys:
				is_new = key not in old_keys
				is_removed = key not in keys
				visible = self.visibility.get(key, True)

				# Ensure group exists
				if is_new:
					group = gfx.Group()
					group.name = f"data-{key}"
					self._scene.add(group)
					self._data_groups[key] = group
				elif is_removed:
					# Clean up removed key's group
					group = self._data_groups.pop(key, None)
					if group:
						try:
							self._scene.remove(group)
						except Exception:
							pass
					continue

				group = self._data_groups[key]
				group.visible = visible

				# Clear children for rebuild (always rebuild when dirty)
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
				x_idx, y_idx, c_idx, s_idx = ax_cfg
				color_hex = COLORS[pp[0] % len(COLORS)]
				do_line = pp[1]
				line_type = pp[2]
				line_width = pp[3]
				do_sym = pp[4]
				sym_type = pp[5]
				sym_size = pp[6]
				alpha = pp[10] if len(pp) > 10 else 0.8

				# Get x and y arrays
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

				# Validate data lengths match
				if len(x) != len(y):
					min_len = min(len(x), len(y))
					x = x[:min_len]
					y = y[:min_len]

				if len(x) == 0:
					continue

				# Apply log transforms if needed
				if self.xlog and np.min(x) > 0:
					x = np.log10(x)
				if self.ylog and np.min(y) > 0:
					y = np.log10(y)

				# Create line object if needed - add to GROUP not scene
				if do_line and len(x) > 1:
					positions = np.column_stack([x, y, np.zeros_like(x)])
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

					line_obj = gfx.Line(gfx.Geometry(positions=positions), line_material)
					group.add(line_obj)

				# Create point markers if needed - add to GROUP not scene
				if do_sym:
					marker_size = max(sym_size, 2)
					marker_nodes = self._create_markers(
						x, y, sym_type, marker_size, color_hex, alpha)
					for mn in marker_nodes:
						group.add(mn)
		finally:
			self._rebuilding = False
		# After rebuild, schedule another render if still dirty or shapes pending
		self._request_render()

	def _create_markers(self, x, y, sym_type, size, color_hex, alpha):
		"""Create marker points. Returns list of nodes (NOT added to scene)."""
		n = len(x)
		if n == 0:
			return []

		positions = np.column_stack([x, y, np.zeros_like(x)])
		marker_color = _hex_to_rgba(color_hex)[:3] + (alpha,)

		# Marker shape mapping: index -> pygfx PointsMarkerMaterial marker name
		shape_map = {
			0: "circle",
			1: "square",
			2: "plus",
			3: "triangle_up",
			4: "triangle_down",
		}
		marker_name = shape_map.get(sym_type, "circle")

		try:
			marker_obj = gfx.Points(
				gfx.Geometry(positions=positions),
				gfx.PointsMarkerMaterial(
					size=size,
					marker=marker_name,
					color=marker_color,
				))
			return [marker_obj]
		except Exception as e:
			print(f"Marker creation failed ({marker_name}): {e}")
			return []

	def _render_plot(self):
		"""Per-frame render - only safe ops (no scene graph modifications)."""
		renderer = self._renderer
		camera = self._camera

		# Get screen dimensions
		w, h = renderer.logical_size[0], renderer.logical_size[1]
		if w > 0 and h > 0:
			camera.set_view_size(w, h)

		margin_left = 65
		margin_bottom = 50
		margin_top = 20
		margin_right = 20

		# Compute current visible world bounds from camera frustum
		try:
			f = camera.frustum  # shape (2, 4, 3): [near|far, corners, xyz]
			near_corners = f[0]  # (4, 3)
			xleft = float(near_corners[:, 0].min())
			xright = float(near_corners[:, 0].max())
			ybottom = float(near_corners[:, 1].min())
			ytop = float(near_corners[:, 1].max())
		except Exception:
			xleft, xright = self.xlimits if self.xlimits else (0, 1)
			ybottom, ytop = self.ylimits if self.ylimits else (0, 1)
		xspan = xright - xleft
		yspan = ytop - ybottom
		frac_left = margin_left / w
		frac_right = margin_right / w
		frac_bottom = margin_bottom / h
		frac_top = margin_top / h

		# Border box corners (world coords)
		bl_x = xleft + xspan * frac_left
		bl_y = ybottom + yspan * frac_bottom
		br_x = xleft + xspan * (1 - frac_right)
		br_y = ybottom + yspan * frac_bottom
		tl_x = xleft + xspan * frac_left
		tl_y = ybottom + yspan * (1 - frac_top)
		tr_x = xleft + xspan * (1 - frac_right)
		tr_y = ybottom + yspan * (1 - frac_top)

		# Position bottom X ruler
		self._ruler_x_bottom.start_pos = (bl_x, bl_y, 5)
		self._ruler_x_bottom.end_pos = (br_x, br_y, 5)
		self._ruler_x_bottom.start_value = bl_x

		# Position top X ruler (tick marks only)
		self._ruler_x_top.start_pos = (tl_x, tl_y, 5)
		self._ruler_x_top.end_pos = (tr_x, tr_y, 5)
		self._ruler_x_top.start_value = tl_x

		# Position left Y ruler
		self._ruler_y_left.start_pos = (bl_x, bl_y, 5)
		self._ruler_y_left.end_pos = (tl_x, tl_y, 5)
		self._ruler_y_left.start_value = bl_y

		# Position right Y ruler (tick marks only)
		self._ruler_y_right.start_pos = (br_x, br_y, 5)
		self._ruler_y_right.end_pos = (tr_x, tr_y, 5)
		self._ruler_y_right.start_value = br_y

		# Update rulers to compute tick spacing
		# Set minimum tick spacing in world units to prevent rulers vanishing at extreme zoom
		try:
			stats_x = self._ruler_x_bottom.update(camera, (w, h))
			stats_y = self._ruler_y_left.update(camera, (w, h))
			self._ruler_x_top.update(camera, (w, h))
			self._ruler_y_right.update(camera, (w, h))

			major_x = max(stats_x.get("tick_step", 1), xspan * 0.01)
			major_y = max(stats_y.get("tick_step", 1), yspan * 0.01)

		except Exception:
			major_x, major_y = xspan * 0.1, yspan * 0.1

		# Grid step size (for reference, though grid is disabled)
		# self._grid.material.major_step = (major_x, major_y)

		# Update border box (positions change with zoom/pan)
		box_pts = np.array([
			[bl_x, bl_y, 5],
			[br_x, br_y, 5],
			[tr_x, tr_y, 5],
			[tl_x, tl_y, 5],
			[bl_x, bl_y, 5],
		], dtype=np.float32)
		if self._border_box:
			try:
				self._scene.remove(self._border_box)
			except Exception:
				pass
		self._border_box = gfx.Line(
			gfx.Geometry(positions=box_pts),
			gfx.LineMaterial(thickness=1.5, color=(0, 0, 0, 1.0)))
		self._scene.add(self._border_box)

		# Render shape overlays on top of data (BEFORE render so they're visible)
		self._render_shapes(w, h)

		# Position axis labels in world coords so they stay fixed on screen
		# Check if label text has changed and needs recreating
		labels_need_update = (len(self._axis_labels) < 2) or (self.xaxis_label != self._prev_xlabel) or (self.yaxis_label != self._prev_ylabel)

		if labels_need_update:
			# Remove old axis labels
			for lbl in self._axis_labels:
				try:
					self._scene.remove(lbl)
				except Exception:
					pass
			self._axis_labels = []

			label_color = (0, 0, 0, 1.0)
			if self.xaxis_label:
				xlbl = gfx.Text(self.xaxis_label, font_size=18,
									screen_space=True)
				xlbl.material.color = label_color
				sx_lbl = margin_left + (w - margin_left - margin_right) / 2
				sy_lbl = h - margin_bottom + 25
				xlbl._screen_pos = (sx_lbl, sy_lbl)
				xlbl.local.position = (*self._screen_to_world(sx_lbl, sy_lbl), 0)
				self._scene.add(xlbl)
				self._axis_labels.append(xlbl)

			if self.yaxis_label:
				ylbl = gfx.Text(self.yaxis_label, font_size=18,
									screen_space=True)
				ylbl.material.color = label_color
				sy_lbl = margin_top + (h - margin_top - margin_bottom) / 2 - 10
				ylbl._screen_pos = (10, sy_lbl)
				ylbl.local.position = (*self._screen_to_world(10, sy_lbl), 0)
				ylbl.local.rotation = la.quat_from_euler((0, 0, np.radians(90)))  # Rotate 90° CCW
				self._scene.add(ylbl)
				self._axis_labels.append(ylbl)

			# Cache values for next comparison
			self._prev_xlabel = self.xaxis_label
			self._prev_ylabel = self.yaxis_label

		for lbl in self._axis_labels:
			sx_lbl, sy_lbl = lbl._screen_pos  # stored screen coords
			wx_lbl, wy_lbl = self._screen_to_world(sx_lbl, sy_lbl)
			lbl.local.position = (wx_lbl, wy_lbl, 0)

		renderer.render(self._scene, camera)

	def _rebuild_if_dirty(self):
		self._dirty = True
		self._request_render()

	def set_data(self, input_data, key="data", replace=False, quiet=False,
				 color=-1, linewidth=1, linetype=-2, symtype=-2, symsize=6,
				 comments=None, column_labels=None):
		"""Set a keyed data set.

		input_data: numpy array (plotted vs index), or list/tuple of arrays
					where each array is one column/axis.
		key: string identifier for this data set.
		replace: if True, clear all existing data sets first.
		quiet: if True, don't trigger immediate re-render.
		color: color index (0-7), -1 for auto-assign.
		linetype: 0=solid, 1=dashed, 2=dotted, 3=dashdot, -1=disable, -2=auto.
		symtype: 0=circle, 1=square, 2=plus, 3=triup, 4=tridown, -1=disable, -2=auto.
		symsize: marker size in pixels.
		column_labels: list of strings naming each column.
		"""
		self._dirty = True

		if replace:
			for key in list(self.data.keys()):
				self.clear_data(key)

		if input_data is None:
			self.clear_data(key)
			return

		# Convert to numpy arrays
		data_list = self._convert_data(input_data)

		if not data_list:
			return

		is_new_key = key not in self.data

		self.data[key] = data_list
		self.visibility.setdefault(key, True)

		# Determine axis mapping
		if is_new_key:
			n_cols = len(data_list)
			if n_cols == 1:
				self.axes[key] = (-1, 0, -2, -2)
			elif n_cols == 2:
				self.axes[key] = (0, 1, -2, -2)
			else:
				self.axes[key] = (0, 1, -2, -2)

		# Determine plot parameters
		if is_new_key:
			# Auto-assign color
			if color < 0:
				color = len(self.data) % len(COLORS)

			# Auto-detect line vs symbol
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
				xa, ya = self.axes[key][:2]
				if xa >= 0 and ya >= 0:
					self.set_axis_parms(
						str(column_labels[xa]), str(column_labels[ya]))
			except Exception:
				pass

		self.autoscale()

		if self.inspector:
			self.inspector.datachange()

		if not quiet:
			self._dirty = True
			self._request_render()

	def clear_data(self, key):
		"""Remove a data set by key."""
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
			if len(input_data) > 0:
				try:
					first = input_data[0]
					if not isinstance(first, (np.ndarray, list, tuple)):
						# Single array of scalars -> plot vs index
						x_axis = np.arange(len(input_data), dtype=np.float32)
						return [x_axis, np.asarray(input_data, dtype=np.float32)]
				except Exception:
					pass

			result = []
			for item in input_data:
				try:
					arr = np.asarray(item, dtype=np.float32).ravel()
					result.append(arr)
				except Exception as e:
					print(f"Data conversion error: {e}")
					return None
			return result

		return None

	def autoscale(self):
		"""Auto-scale axes to fit all visible data."""
		all_x = []
		all_y = []

		for key, data_list in self.data.items():
			if not self.visibility.get(key, True):
				continue
			ax_cfg = self.axes.get(key)
			if ax_cfg is None:
				continue

			x_idx, y_idx = ax_cfg[:2]
			try:
				if x_idx == -1:
					x = np.arange(len(data_list[y_idx]), dtype=np.float32)
				elif x_idx < len(data_list):
					x = data_list[x_idx]
				else:
					continue

				if y_idx < len(data_list):
					y = data_list[y_idx]
				else:
					continue

				n = min(len(x), len(y))
				all_x.append(x[:n])
				all_y.append(y[:n])
			except Exception:
				pass

		if not all_x or not all_y:
			return

		x_all = np.concatenate(all_x)
		y_all = np.concatenate(all_y)

		# Remove inf/nan for range computation
		mask = np.isfinite(x_all) & np.isfinite(y_all)
		x_finite = x_all[mask]
		y_finite = y_all[mask]

		if len(x_finite) == 0:
			return

		x_range = x_finite.max() - x_finite.min()
		y_range = y_finite.max() - y_finite.min()

		padding_x = max(x_range * 0.05, 1e-10)
		padding_y = max(y_range * 0.05, 1e-10)

		self.xlimits = (x_finite.min() - padding_x, x_finite.max() + padding_x)
		self.ylimits = (y_finite.min() - padding_y, y_finite.max() + padding_y)
		self._update_camera_limits()

		# Update inspector limit boxes if open
		if self.inspector:
			self.inspector._update_limits_from_widget()

	def set_axis_parms(self, xlabel="", ylabel="", x_scale="linear",
					   y_scale="linear"):
		"""Set axis labels and scale type."""
		self.xaxis_label = xlabel
		self.yaxis_label = ylabel
		self.xlog = (x_scale == "log")
		self.ylog = (y_scale == "log")

	def set_xlimits(self, xmin, xmax):
		self.xlimits = (xmin, xmax)
		if self.ylimits is not None:
			self._update_camera_limits()
		self._dirty = True
		self._request_render()

	def set_ylimits(self, ymin, ymax):
		self.ylimits = (ymin, ymax)
		if self.xlimits is not None:
			self._update_camera_limits()
		self._dirty = True
		self._request_render()

	def full_refresh(self):
		self._dirty = True
		self._request_render()

	def get_inspector(self):
		if self.inspector is None:
			self.inspector = EMPlot2DInspector(self)
		return self.inspector

	def show_inspector(self, show=True):
		if show:
			insp = self.get_inspector()
			insp.show()
			insp.raise_()
			insp.activateWindow()
			self._dirty = True
			self._request_render()
		else:
			if self.inspector:
				self.inspector.close()

	# Mouse event handlers

	def _render_shapes(self, w, h):
		"""Render shape overlays on top of data."""
		if self._shape_group is None:
			self._shape_group = gfx.Group()
			self._shape_group.render_order = 50
			self._scene.add(self._shape_group)

		# Only rebuild if shapes actually changed (avoids continuous updates when mouse still)
		if not self.shapes and not self._shape_group.children:
			return
		if len(self.shapes) == 0 and len(self._shape_group.children) > 0:
			while len(self._shape_group.children) > 0:
				self._shape_group.remove(self._shape_group.children[0])
			return
		if len(self.shapes) == 0:
			return

		# Check if any shape data actually changed since last frame
		shapes_changed = False
		for name, shape_data in self.shapes.items():
			if name not in self._prev_shapes or self._prev_shapes[name] != shape_data:
				shapes_changed = True
				break
		# Also check for removed shapes
		if set(self.shapes.keys()) != set(self._prev_shapes.keys()):
			shapes_changed = True

		if not shapes_changed:
			return

		# Save current shapes for next comparison
		self._prev_shapes = dict(self.shapes)

		# Clear old shapes and rebuild
		while len(self._shape_group.children) > 0:
			self._shape_group.remove(self._shape_group.children[0])

		for name, shape_data in self.shapes.items():
			try:
				stype = shape_data.get("type", "")
				color = shape_data.get("color", (0.0, 0.0, 0.0))
				lw = shape_data.get("lw", 1.0)

				if stype == "line":
					sx0, sy0 = shape_data["sx0"], shape_data["sy0"]
					sx1, sy1 = shape_data["sx1"], shape_data["sy1"]
					wx0, wy0 = self._screen_to_world(sx0, sy0)
					wx1, wy1 = self._screen_to_world(sx1, sy1)
					verts = np.array([[wx0, wy0, 0.1], [wx1, wy1 , 0.1]], dtype=np.float32)
					node = gfx.Line(gfx.Geometry(positions=verts),
					   gfx.LineMaterial(thickness=lw, color=(color[0], color[1], color[2], 1.0)))
					node.render_order = 999
					self._shape_group.add(node)

				elif stype == "rect":
					sx0, sy0 = shape_data["sx0"], shape_data["sy0"]
					sx1, sy1 = shape_data["sx1"], shape_data["sy1"]
					wx0, wy0 = self._screen_to_world(sx0, sy0)
					wx1, wy1 = self._screen_to_world(sx1, sy1)
					verts = np.array([
						[wx0, wy0, 0.1], [wx1, wy0, 0.1],
						[wx1, wy1, 0.1], [wx0, wy1, 0.1],
						[wx0, wy0, 0.1]], dtype=np.float32)
					node = gfx.Line(gfx.Geometry(positions=verts),
					   gfx.LineMaterial(thickness=lw, color=(color[0], color[1], color[2], 1.0)))
					node.render_order = 999
					self._shape_group.add(node)

				elif stype == "label":
					sx_val = shape_data.get("sx", 100)
					sy_val = shape_data.get("sy", 50)
					fs = shape_data.get("font_size", 14)
					node = gfx.Text(str(shape_data["text"]), font_size=fs,
					   anchor='middle-center', screen_space=True, render_order=999)
					node.local.position = (*self._screen_to_world(sx_val,sy_val), 0)
					node.material.color = (color[0], color[1], color[2], 1.0)
					self._shape_group.add(node)

			except Exception as e:
				print(f"Shape render error: {name}: {e}")

	def _on_mouse_press(self, event):
		sx, sy = event.position().x(), event.position().y()
		w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
		world_pos = self._screen_to_world(sx, sy)

		if self.mouseemit:
			self.mousedown.emit(event, world_pos)

		# Middle click -> show inspector
		if event.button() == Qt.MouseButton.MiddleButton or \
		   (event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.AltModifier):
			self.show_inspector(True)
			return

			# Right click -> start rubber-band zoom box
		if event.button() == Qt.MouseButton.RightButton:
			self.shapes = {}
			self._right_drag_pos = (sx, sy)
			self._request_render()
			return

		# Left click -> crosshair probe mode
		if event.button() == Qt.MouseButton.LeftButton:
			self._left_drag_active = True
			margin_left = 65
			margin_right = 20
			margin_top = 20
			# Horizontal crosshair (screen coords) - black for visibility on white bg
			hx1, hy1 = margin_left, sy
			hx2, hy2 = w - margin_right, sy
			self.shapes["xcross"] = {"type": "line", "color": (0.0, 0.0, 0.0),
			                         "lw": 1.0, "sx0": hx1, "sy0": hy1, "sx1": hx2, "sy1": hy2}
			# Vertical crosshair (screen coords) - full height of widget
			vx1, vy1 = sx, 0
			vx2, vy2 = sx, h
			self.shapes["ycross"] = {"type": "line", "color": (0.0, 0.0, 0.0),
			                         "lw": 1.0, "sx0": vx1, "sy0": vy1, "sx1": vx2, "sy1": vy2}
			label_text = "(%g, %g)" % (world_pos[0], world_pos[1])
			self.shapes["coord"] = {"type": "label", "color": (0.0, 0.0, 0.0),
		                            "sx": w - margin_right - 100, "sy": margin_top + 30,
		                            "text": label_text, "font_size": 15}
			self._request_render()

	def _on_mouse_move(self, event):
		sx, sy = event.position().x(), event.position().y()
		w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
		world_pos = self._screen_to_world(sx, sy)

		if self.mouseemit and self._left_drag_active:
			self.mousedrag.emit(event, world_pos)

		# Right-drag -> rubber-band zoom box (screen coords) - black
		if self._right_drag_pos is not None:
			x1r, y1r = self._right_drag_pos
			self.shapes["zoombox"] = {"type": "rect", "color": (0.0, 0.0, 0.0),
		                          "lw": 1.5, "sx0": x1r, "sy0": y1r,
		                          "sx1": sx, "sy1": sy}
			self._request_render()
			return

		# Left-drag -> update crosshairs + coord label
		if self._left_drag_active:
			margin_left = 65
			margin_right = 20
			margin_top = 20
			hx1, hy1 = margin_left, sy
			hx2, hy2 = w - margin_right, sy
			self.shapes["xcross"] = {"type": "line", "color": (0.0, 0.0, 0.0),
			                         "lw": 1.0, "sx0": hx1, "sy0": hy1, "sx1": hx2, "sy1": hy2}
			vx1, vy1 = sx, 0
			vx2, vy2 = sx, h
			self.shapes["ycross"] = {"type": "line", "color": (0.0, 0.0, 0.0),
			                         "lw": 1.0, "sx0": vx1, "sy0": vy1, "sx1": vx2, "sy1": vy2}
			label_text = "(%g, %g)" % (world_pos[0], world_pos[1])
			self.shapes["coord"] = {"type": "label", "color": (0.0, 0.0, 0.0),
		                            "sx": w - margin_right - 100, "sy": margin_top + 30,
		                            "text": label_text, "font_size": 15}
			self._request_render()

	def _on_mouse_release(self, event):
		sx, sy = event.position().x(), event.position().y()

		# Right release -> apply rubber-band zoom or rescale
		if self._right_drag_pos is not None:
			self.shapes = {}  # clear overlay shapes
			x1r, y1r = self._right_drag_pos
			dx = abs(sx - x1r) + abs(sy - y1r)
			if dx < 3:
				self.autoscale()
			else:
				p1 = self._screen_to_world(x1r, y1r)
				p2 = self._screen_to_world(sx, sy)
				xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
				ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
				if xmin < xmax and ymin < ymax:
					self.xlimits = (xmin, xmax)
					self.ylimits = (ymin, ymax)
				else:
					self.autoscale()
			try:
				self._update_camera_limits()
			except Exception as e:
				pass
			self._right_drag_pos = None
			self._dirty = True
			self._request_render()
			# Update inspector limit boxes if open
			if self.inspector:
				self.inspector._update_limits_from_widget()
			return

		# Left release -> clear crosshairs
		if self._left_drag_active:
			self._dirty = True
			self.shapes = {}
			self._left_drag_active = False
			self._request_render()

		if self.mouseemit and event.button() == Qt.MouseButton.LeftButton:
			world_pos = self._screen_to_world(sx, sy)
			self.mouseup.emit(event, world_pos)

	def _on_wheel(self, event):
		"""Scroll wheel zooms about cursor position."""
		sx, sy = event.position().x(), event.position().y()
		world_pos = self._screen_to_world(sx, sy)
		xc, yc = world_pos

		if self.xlimits is None or self.ylimits is None:
			self.autoscale()
			self._dirty = True
			self._request_render()
			return

		xleft, xright = self.xlimits
		ybottom, ytop = self.ylimits
		xfrac = (xc - xleft) / (xright - xleft) if xright != xleft else 0.5
		yfrac = (yc - ybottom) / (ytop - ybottom) if ytop != ybottom else 0.5

		delta = event.angleDelta().y()
		if delta > 0:
			zoom_factor = 1.2  # zoom in
		else:
			zoom_factor = 1.0 / 1.2  # zoom out

		xspan = (xright - xleft) / zoom_factor
		ypan = (ytop - ybottom) / zoom_factor
		self.xlimits = (xc - xfrac * xspan, xc + (1 - xfrac) * xspan)
		self.ylimits = (yc - yfrac * ypan, yc + (1 - yfrac) * ypan)

		self._update_camera_limits()
		self._dirty = True
		self._request_render()
		# Update inspector limit boxes if open
		if self.inspector:
			self.inspector._update_limits_from_widget()

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



# Map color index to display name for list items
COLOR_NAMES = ["Black", "Blue", "Red", "Green", "Cyan", "Magenta", "Yellow", "Grey"]

# RGB values for color indices (used to color the list widget text)
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


class EMPlot2DInspector(QtWidgets.QWidget):

	def __init__(self, target_widget):
		super().__init__()
		self.target = weakref.ref(target_widget)
		self.setWindowTitle("Plot Controls")
		self.resize(400, 500)

		self._build_ui()
		# Wire signals (without limit signals) before sync
		self._wire_signals()
		self._sync_from_widget()
		# Connect limit signals AFTER sync so setValue doesn't trigger changes with partial data
		self.xmin_box.valueChanged.connect(self._on_limit_change)
		self.xmax_box.valueChanged.connect(self._on_limit_change)
		self.ymin_box.valueChanged.connect(self._on_limit_change)
		self.ymax_box.valueChanged.connect(self._on_limit_change)

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
		dsl.addWidget(self.setlist)

		h_sel = QtWidgets.QHBoxLayout()
		self.none_but = QtWidgets.QPushButton("None")
		self.none_but.clicked.connect(self.sel_none)
		h_sel.addWidget(self.none_but)
		self.all_but = QtWidgets.QPushButton("All")
		self.all_but.clicked.connect(self.sel_all)
		h_sel.addWidget(self.all_but)
		dsl.addLayout(h_sel)

		vbl.addWidget(dsg)

		# ── Save / Analysis buttons ──
		btn_row = QtWidgets.QHBoxLayout()
		self.save_btn = QtWidgets.QPushButton("Save")
		self.save_btn.clicked.connect(self._on_save_plot)
		btn_row.addWidget(self.save_btn)
		self.stats_btn = QtWidgets.QPushButton("Statistics")
		self.stats_btn.clicked.connect(self._on_statistics)
		btn_row.addWidget(self.stats_btn)
		self.regress_btn = QtWidgets.QPushButton("Regression")
		self.regress_btn.clicked.connect(self._on_regression)
		btn_row.addWidget(self.regress_btn)
		vbl.addLayout(btn_row)

		# ── Appearance controls ──
		apg = QtWidgets.QGroupBox("Appearance")
		apl = QtWidgets.QVBoxLayout(apg)

		self.color_combo = QtWidgets.QComboBox()
		self.color_combo.addItems(COLOR_NAMES)
		h_color = QtWidgets.QHBoxLayout()
		h_color.addWidget(QtWidgets.QLabel("Color:"))
		h_color.addWidget(self.color_combo)
		apl.addLayout(h_color)

		# Line/Symbol toggles
		toggle_row = QtWidgets.QHBoxLayout()
		self.line_tog = QtWidgets.QPushButton("Line")
		self.line_tog.setCheckable(True)
		self.line_tog.clicked.connect(self._on_appearance_change)
		self.sym_tog = QtWidgets.QPushButton("Symbol")
		self.sym_tog.setCheckable(True)
		self.sym_tog.clicked.connect(self._on_appearance_change)
		toggle_row.addWidget(self.line_tog)
		toggle_row.addWidget(self.sym_tog)
		apl.addLayout(toggle_row)

		# Line type and width
		line_row = QtWidgets.QHBoxLayout()
		line_row.addWidget(QtWidgets.QLabel("Line:"))
		self.linetype_combo = QtWidgets.QComboBox()
		self.linetype_combo.addItems(["Solid", "Dashed", "Dotted", "Dash-Dot"])
		line_row.addWidget(self.linetype_combo)
		self.linewidth_spin = QtWidgets.QSpinBox()
		self.linewidth_spin.setRange(1, 10)
		line_row.addWidget(QtWidgets.QLabel("Width:"))
		line_row.addWidget(self.linewidth_spin)
		apl.addLayout(line_row)

		# Symbol type and size
		sym_row = QtWidgets.QHBoxLayout()
		sym_row.addWidget(QtWidgets.QLabel("Symbol:"))
		self.symtype_combo = QtWidgets.QComboBox()
		self.symtype_combo.addItems(["Circle", "Square", "Plus", "TriUp", "TriDown"])
		sym_row.addWidget(self.symtype_combo)
		self.symsize_spin = QtWidgets.QSpinBox()
		self.symsize_spin.setRange(1, 30)
		sym_row.addWidget(QtWidgets.QLabel("Size:"))
		sym_row.addWidget(self.symsize_spin)
		apl.addLayout(sym_row)

		vbl.addWidget(apg)

		# ── Column selectors ──
		cg = QtWidgets.QGroupBox("Columns")
		cl = QtWidgets.QGridLayout()
		cl.addWidget(QtWidgets.QLabel("X:"), 0, 0)
		self.col_x_spin = QtWidgets.QSpinBox()
		self.col_x_spin.setRange(-1, 5)
		cl.addWidget(self.col_x_spin, 0, 1)
		cl.addWidget(QtWidgets.QLabel("Y:"), 0, 2)
		self.col_y_spin = QtWidgets.QSpinBox()
		self.col_y_spin.setRange(-1, 5)
		cl.addWidget(self.col_y_spin, 0, 3)
		cg.setLayout(cl)
		vbl.addWidget(cg)

		# ── Axis scale toggles ──
		scale_row = QtWidgets.QHBoxLayout()
		self.xlog_tog = QtWidgets.QPushButton("X Log")
		self.xlog_tog.setCheckable(True)
		scale_row.addWidget(self.xlog_tog)
		self.ylog_tog = QtWidgets.QPushButton("Y Log")
		self.ylog_tog.setCheckable(True)
		scale_row.addWidget(self.ylog_tog)
		self.rescale_btn = QtWidgets.QPushButton("Rescale")
		scale_row.addWidget(self.rescale_btn)
		vbl.addLayout(scale_row)

		# ── Axis limits ──
		limit_g = QtWidgets.QGridLayout()
		self.xmin_box = ValBox(label="X:", value=0)
		self.xmax_box = ValBox(label="", value=1)
		self.ymin_box = ValBox(label="Y:", value=0)
		self.ymax_box = ValBox(label="", value=1)
		limit_g.addWidget(self.xmin_box, 0, 0)
		limit_g.addWidget(self.xmax_box, 0, 1)
		limit_g.addWidget(self.ymin_box, 1, 0)
		limit_g.addWidget(self.ymax_box, 1, 1)
		vbl.addLayout(limit_g)

		# ── Labels only (no title) ──
		label_g = QtWidgets.QGridLayout()
		self.xlabel_edit = QtWidgets.QLineEdit()
		self.ylabel_edit = QtWidgets.QLineEdit()
		label_g.addWidget(QtWidgets.QLabel("X Label:"), 0, 0)
		label_g.addWidget(self.xlabel_edit, 0, 1)
		label_g.addWidget(QtWidgets.QLabel("Y Label:"), 1, 0)
		label_g.addWidget(self.ylabel_edit, 1, 1)
		vbl.addLayout(label_g)

		# ── Transparency slider ──
		self.alpha_slider = ValSlider(label="Alpha", value=0.8)
		self.alpha_slider.setRange(0.1, 1.0)
		vbl.addWidget(self.alpha_slider)

		vbl.addStretch()

	def _wire_signals(self):
		"""Connect all control signals to handlers."""
		tgt = self._get_tgt()
		if not tgt:
			return

		# Data set list selection -> populate controls
		self.setlist.currentRowChanged.connect(self._on_selection_changed)

		# Color combo -> change color for selected dataset
		self.color_combo.currentIndexChanged.connect(self._on_color_change)

		# Line type / width
		self.linetype_combo.currentIndexChanged.connect(self._on_appearance_change)
		self.linewidth_spin.valueChanged.connect(self._on_appearance_change)

		# Symbol type / size
		self.symtype_combo.currentIndexChanged.connect(self._on_appearance_change)
		self.symsize_spin.valueChanged.connect(self._on_appearance_change)

		# Column selectors
		self.col_x_spin.valueChanged.connect(self._on_column_change)
		self.col_y_spin.valueChanged.connect(self._on_column_change)

		# Axis log toggles
		self.xlog_tog.clicked.connect(self._on_scale_change)
		self.ylog_tog.clicked.connect(self._on_scale_change)

		# Rescale button
		self.rescale_btn.clicked.connect(self._on_rescale)

		# Axis limits (connected in __init__ AFTER sync to avoid setValue interference)
		# xmin_box.valueChanged.connect(self._on_limit_change)
		# xmax_box.valueChanged.connect(self._on_limit_change)
		# ymin_box.valueChanged.connect(self._on_limit_change)
		# ymax_box.valueChanged.connect(self._on_limit_change)

		# Labels and title

		# Labels
		self.xlabel_edit.editingFinished.connect(self._on_label_change)
		self.ylabel_edit.editingFinished.connect(self._on_label_change)

		# Alpha slider
		self.alpha_slider.valueChanged.connect(self._on_alpha_change)

	def _sync_from_widget(self):
		"""Sync inspector controls to match widget state on open."""
		tgt = self._get_tgt()
		if not tgt:
			return

		# Axis limits
		if tgt.xlimits is not None:
			self.xmin_box.setValue(tgt.xlimits[0])
			self.xmax_box.setValue(tgt.xlimits[1])
		if tgt.ylimits is not None:
			self.ymin_box.setValue(tgt.ylimits[0])
			self.ymax_box.setValue(tgt.ylimits[1])

		# Labels
		self.xlabel_edit.setText(tgt.xaxis_label)
		self.ylabel_edit.setText(tgt.yaxis_label)

		# Log scale toggles
		self.xlog_tog.setChecked(tgt.xlog)
		self.ylog_tog.setChecked(tgt.ylog)

		# Rebuild list and select first item
		self.update_list()

	def _update_limits_from_widget(self):
		"""Update limit boxes from widget without triggering _on_limit_change."""
		tgt = self._get_tgt()
		if not tgt or tgt.xlimits is None or tgt.ylimits is None:
			return
		# Disconnect signals to avoid partial updates triggering changes
		self.xmin_box.valueChanged.disconnect(self._on_limit_change)
		self.xmax_box.valueChanged.disconnect(self._on_limit_change)
		self.ymin_box.valueChanged.disconnect(self._on_limit_change)
		self.ymax_box.valueChanged.disconnect(self._on_limit_change)
		# Set values
		self.xmin_box.setValue(tgt.xlimits[0])
		self.xmax_box.setValue(tgt.xlimits[1])
		self.ymin_box.setValue(tgt.ylimits[0])
		self.ymax_box.setValue(tgt.ylimits[1])
		# Reconnect
		self.xmin_box.valueChanged.connect(self._on_limit_change)
		self.xmax_box.valueChanged.connect(self._on_limit_change)
		self.ymin_box.valueChanged.connect(self._on_limit_change)
		self.ymax_box.valueChanged.connect(self._on_limit_change)

	def _update_controls_for_key(self, key):
		"""Populate controls with parameters of the selected data set."""
		tgt = self._get_tgt()
		if not tgt or key is None:
			return

		pp = tgt.pparm.get(key)
		if pp is None:
			return

		# Color
		color_idx = pp[0] % len(COLORS)
		self.color_combo.setCurrentIndex(color_idx)

		# Line toggle, type, width
		self.line_tog.setChecked(bool(pp[1]))
		lt = min(pp[2], 3) if pp[2] >= 0 else 0
		self.linetype_combo.setCurrentIndex(lt)
		self.linewidth_spin.setValue(max(1, pp[3]))

		# Symbol toggle, type, size
		self.sym_tog.setChecked(bool(pp[4]))
		st = min(pp[5], 4) if pp[5] >= 0 else 0
		self.symtype_combo.setCurrentIndex(st)
		self.symsize_spin.setValue(max(1, pp[6]))

		# Alpha
		alpha = pp[10] if len(pp) > 10 else 0.8
		self.alpha_slider.setValue(alpha)

		# Column selectors
		ax_cfg = tgt.axes.get(key, (-1, 0, -2, -2))
		self.col_x_spin.setValue(max(-1, min(ax_cfg[0], 5)))
		self.col_y_spin.setValue(max(-1, min(ax_cfg[1], 5)))

	def _on_selection_changed(self, row):
		"""Called when the user selects a data set in the list."""
		item = self.setlist.item(row)
		if item is None:
			return
		key = item.data(QtCore.Qt.UserRole)
		self._update_controls_for_key(key)

	def _selected_key(self):
		"""Get the currently selected data set key."""
		current_row = self.setlist.currentRow()
		if current_row < 0:
			return None
		item = self.setlist.item(current_row)
		if item is None:
			return None
		return item.data(QtCore.Qt.UserRole)

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

	def update_list(self):
		"""Rebuild the data set list, preserving selection."""
		tgt = self._get_tgt()
		if not tgt:
			return

		# Remember current selection
		old_key = self._selected_key()

		self.setlist.clear()
		for key in tgt.data:
			pp = tgt.pparm.get(key)
			color_idx = pp[0] % len(COLORS) if pp else 0
			item_text = key
			if color_idx >= 0 and color_idx < len(COLOR_NAMES):
				item_text = f"{COLOR_NAMES[color_idx]} - {key}"

			item = QtWidgets.QListWidgetItem(item_text)
			item.setData(QtCore.Qt.UserRole, key)

			if not tgt.visibility.get(key, True):
				item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)

			self.setlist.addItem(item)

		self.setlist.repaint()

		# Try to re-select the previously selected item
		if old_key is not None:
			for i in range(self.setlist.count()):
				it = self.setlist.item(i)
				if it and it.data(QtCore.Qt.UserRole) == old_key:
					self.setlist.setCurrentRow(i)
					break

	def datachange(self):
		self.update_list()

	def _on_save_plot(self):
		tgt = self._get_tgt()
		if tgt:
			pass  # Implement later

	def _on_statistics(self):
		tgt = self._get_tgt()
		if tgt:
			pass  # Implement later

	def _on_regression(self):
		tgt = self._get_tgt()
		if tgt:
			pass  # Implement later

	def _on_appearance_change(self):
		key = self._selected_key()
		if key is None or not self.target():
			return
		pp = self.target().pparm.get(key)
		if pp is None:
			return

		pp = list(pp)
		pp[1] = self.line_tog.isChecked()       # do_line
		pp[2] = self.linetype_combo.currentIndex()  # line_type
		pp[3] = max(1, self.linewidth_spin.value())  # line_width
		pp[4] = self.sym_tog.isChecked()          # do_sym
		pp[5] = self.symtype_combo.currentIndex()   # sym_type
		pp[6] = max(1, self.symsize_spin.value())    # sym_size
		self.target().pparm[key] = pp
		tgt = self.target()
		if tgt:
			tgt._dirty = True
			tgt._request_render()

	def _on_color_change(self, idx):
		key = self._selected_key()
		if key is None or not self.target():
			return
		pp = self.target().pparm.get(key)
		if pp is None:
			return
		pp = list(pp)
		pp[0] = idx % len(COLORS)
		self.target().pparm[key] = pp

		# Update the list display to reflect new color
		tgt = self._get_tgt()
		if tgt:
			self.update_list()

		tgt._dirty = True
		if tgt:
			tgt._request_render()

	def _on_column_change(self):
		key = self._selected_key()
		if key is None or not self.target():
			return
		tgt = self.target()

		x_idx = self.col_x_spin.value()
		y_idx = self.col_y_spin.value()

		ax = tgt.axes.get(key)
		if ax:
			ax_cfg = list(ax)
			ax_cfg[0] = x_idx
			ax_cfg[1] = y_idx
			tgt.axes[key] = tuple(ax_cfg)

		tgt.autoscale()
		tgt._dirty = True
		tgt._request_render()

	def _on_scale_change(self):
		tgt = self._get_tgt()
		if tgt:
			tgt.xlog = self.xlog_tog.isChecked()
			tgt.ylog = self.ylog_tog.isChecked()
			tgt._dirty = True
			tgt._request_render()

	def _on_rescale(self):
		tgt = self._get_tgt()
		if tgt:
			tgt.autoscale()
			tgt._dirty = True
			tgt._request_render()

			# Update limit boxes to reflect new auto-scaled range
			if tgt.xlimits is not None:
				self.xmin_box.setValue(tgt.xlimits[0])
				self.xmax_box.setValue(tgt.xlimits[1])
			if tgt.ylimits is not None:
				self.ymin_box.setValue(tgt.ylimits[0])
				self.ymax_box.setValue(tgt.ylimits[1])

	def _on_limit_change(self):
		tgt = self._get_tgt()
		if tgt:
			try:
				xmin = float(self.xmin_box.getValue())
				xmax = float(self.xmax_box.getValue())
				ymin = float(self.ymin_box.getValue())
				ymax = float(self.ymax_box.getValue())
				# Validate ranges before applying
				if xmin >= xmax or ymin >= ymax:
					return
				# Set both limits atomically to avoid camera depth issues
				tgt.xlimits = (xmin, xmax)
				tgt.ylimits = (ymin, ymax)
				try:
					tgt._camera.show_rect(xmin, xmax, ymax, ymin)
				except Exception:
					pass  # Camera may reject extreme aspect ratios
				tgt._dirty = True
				tgt._request_render()
			except (ValueError, TypeError):
				pass

	def _on_label_change(self, new_text=None):
		tgt = self._get_tgt()
		if tgt:
			tgt.xaxis_label = self.xlabel_edit.text()
			tgt.yaxis_label = self.ylabel_edit.text()
			tgt._dirty = True
			tgt._request_render()

	def _on_alpha_change(self, val):
		key = self._selected_key()
		if key is None or not self.target():
			return
		pp = self.target().pparm.get(key)
		if pp is None:
			return
		pp = list(pp)
		pp[10] = val
		self.target().pparm[key] = pp
		tgt = self.target()
		if tgt:
			tgt._dirty = True
			tgt._request_render()

	def closeEvent(self, event):
		tgt = self._get_tgt()
		if tgt:
			tgt.inspector = None
		super().closeEvent(event)
