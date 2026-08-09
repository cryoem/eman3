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

		# Contour plot params
		self.contour_enabled = False
		self.contour_bins = 100
		self.contour_levels = 15

		# Selection
		self.selectpoints = True
		self.selected = []
		self.mouseemit = False

		# Mouse state
		self._right_drag_pos = None
		self._left_drag_active = False
		self._shape_group = None
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
		self._redraw = False  # lightweight redraw flag for annotation-only changes
		self._rebuilding = False
		self._pending_rebuild = False
		self._data_groups = {}
		self._contour_group = None

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
		"""Update camera so data limits land at the ruler corners, not widget edges."""
		if self.xlimits is not None and self.ylimits is not None:
			xmin, xmax = self.xlimits
			ymin, ymax = self.ylimits
			# Validate that limits span a real area before updating camera
			if xmin < xmax and ymin < ymax:
				try:
					w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
					margin_left = 65; margin_right = 20
					margin_top = 20; margin_bottom = 50
					# Use default aspect if renderer not yet sized
					if w <= 0:
						w, h = 800, 600
					xfrac = (margin_left + margin_right) / w
					yfrac = (margin_top + margin_bottom) / h
					new_xspan = (xmax - xmin) / max(1.0 - xfrac, 1e-12)
					new_xmin = xmin - (margin_left / w) * new_xspan
					new_ysspan = (ymax - ymin) / max(1.0 - yfrac, 1e-12)
					new_ymin = ymin - (margin_bottom / h) * new_ysspan
					self._camera.show_rect(new_xmin, new_xmin + new_xspan,
					                       new_ymin + new_ysspan, new_ymin, depth=20)
				except Exception:
					pass  # show_rect failed - leave limits as-is

	def _request_render(self):
		"""Request a render frame - fires when data needs rebuilding or annotations changed."""
		if self._canvas is None:
			return  # renderer not ready yet
		if self._dirty or self._redraw:
			self._canvas.request_draw(self._render_callback)

	def _render_callback(self):
		"""Render frame - called only when something changed."""
		try:
			self._redraw = False

			# Defer scene graph modifications to outside the render callback
			if self._dirty and not self._rebuilding and not self._pending_rebuild:
				self._rebuilding = True
				self._dirty = False
				# Schedule rebuild in next event-loop tick (after rendering)
				self._pending_rebuild = True
				QtCore.QTimer.singleShot(0, self._rebuild_data_groups)

			self._render_plot()

			# Only schedule another frame if there's still pending work
			if self._dirty or self._redraw:
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

			# Compute display labels with log suffix if active
			x_lbl = self.xaxis_label
			if self.xlog and x_lbl:
				x_lbl = x_lbl + " (log10)"
			elif not x_lbl and self.xlog:
				x_lbl = "X (log10)"
			y_lbl = self.yaxis_label
			if self.ylog and y_lbl:
				y_lbl = y_lbl + " (log10)"
			elif not y_lbl and self.ylog:
				y_lbl = "Y (log10)"

			if x_lbl:
				xlbl = gfx.Text(x_lbl, font_size=18,
									screen_space=True)
				xlbl.material.color = label_color
				sx_lbl = margin_left + (w - margin_left - margin_right) / 2
				sy_lbl = h - margin_bottom + 25
				xlbl._screen_pos = (sx_lbl, sy_lbl)
				xlbl.local.position = (*self._screen_to_world(sx_lbl, sy_lbl), 0)
				self._scene.add(xlbl)
				self._axis_labels.append(xlbl)

			if y_lbl:
				ylbl = gfx.Text(y_lbl, font_size=18,
									screen_space=True)
				ylbl.material.color = label_color
				sy_lbl = margin_top + (h - margin_top - margin_bottom) / 2 - 10
				ylbl._screen_pos = (10, sy_lbl)
				ylbl.local.position = (*self._screen_to_world(10, sy_lbl), 0)
				ylbl.local.rotation = la.quat_from_euler((0, 0, np.radians(90)))  # Rotate 90° CCW
				self._scene.add(ylbl)
				self._axis_labels.append(ylbl)

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

		# Re-apply camera limits now that renderer has real size
		if self.xlimits and self.ylimits:
			self._update_camera_limits()

		# Trigger final render after full scene initialization
		self._dirty = True
		self._redraw = True
		self._request_render()

	def _rebuild_data_groups(self):
		"""Rebuild data groups - called outside render callback (timer or direct)."""
		if self._pending_rebuild:
			self._pending_rebuild = False
		self._rebuilding = True
		try:
			# Process all data sets - rebuild only changed or new keys
			keys = list(self.data.keys())
			old_keys = list(self._data_groups.keys())
			all_keys = list(dict.fromkeys(keys + old_keys))

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
				x_idx, y_idx = ax_cfg
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

			# Reorder scene children to match list insertion order (last loaded = on top)
			self._reorder_data_groups()
			# Render contours after data groups
			self._render_contours()
		finally:
			self._rebuilding = False
		# Mark as needing a final render frame after rebuild completes
		self._redraw = True
		self._request_render()

	def _reorder_data_groups(self):
		"""Reorder scene children so data groups appear in list insertion order.

		First-loaded datasets render at the bottom, last-loaded on top.
		Non-data-group scene children stay in their original relative order.
		"""
		data_order = list(self.data.keys())
		new_children = []
		for child in self._scene.children:
			if child not in self._data_groups.values():
				new_children.append(child)

		# Append data groups in insertion order (first at bottom, last on top)
		for key in data_order:
			group = self._data_groups.get(key)
			if group is not None and group not in new_children:
				new_children.append(group)

		# Only modify scene if order actually changed
		current = list(self._scene.children)
		if current != new_children:
			while len(self._scene.children) > 0:
				self._scene.remove(self._scene.children[0])
			for child in new_children:
				self._scene.add(child)

	def _render_contours(self):
		"""Build contour lines from visible point data using matplotlib contour extraction."""
		if not self.contour_enabled or self.xlimits is None or self.ylimits is None:
			# Remove existing contours if disabled
			if self._contour_group is not None:
				self._scene.remove(self._contour_group)
				self._contour_group = None
			return

		# Collect all visible point data (already log-transformed if needed)
		all_x, all_y = [], []
		for key, data_list in self.data.items():
			if not self.visibility.get(key, True):
				continue
			ax_cfg = self.axes.get(key)
			if ax_cfg is None:
				continue
			x_idx, y_idx = ax_cfg
			try:
				if x_idx == -1:
					x = np.arange(len(data_list[y_idx]), dtype=np.float64)
				elif x_idx < len(data_list):
					x = np.asarray(data_list[x_idx], dtype=np.float64)
				else:
					continue
				if y_idx < len(data_list):
					y = np.asarray(data_list[y_idx], dtype=np.float64)
				else:
					continue
				n = min(len(x), len(y))
				xx, yy = x[:n], y[:n]
				# Apply log transforms to match what _rebuild_data_groups renders
				if self.xlog and np.min(xx) > 0:
					xx = np.log10(xx)
				if self.ylog and np.min(yy) > 0:
					yy = np.log10(yy)
				all_x.append(xx)
				all_y.append(yy)
			except Exception:
				pass

		if not all_x or len(all_x[0]) == 0:
			if self._contour_group is not None:
				self._scene.remove(self._contour_group)
				self._contour_group = None
			return

		x_all = np.concatenate(all_x)
		y_all = np.concatenate(all_y)

		# Create 2D histogram (density mesh) using current plot limits as boundaries
		try:
			xmin, xmax = self.xlimits
			ymin, ymax = self.ylimits
			bins = self.contour_bins
			hist, xedges, yedges = np.histogram2d(
				x_all, y_all,
				bins=[bins, bins],
				range=[[xmin, xmax], [ymin, ymax]])
			# hist is (ny, nx) with y increasing upwards
		except Exception:
			if self._contour_group is not None:
				self._scene.remove(self._contour_group)
				self._contour_group = None
			return

		# Compute contour levels from mesh min/max (skip zeros)
		nz = hist[hist > 0]
		if len(nz) == 0:
			return
		mesh_min, mesh_max = float(nz.min()), float(nz.max())
		if mesh_min == mesh_max:
			mesh_min = max(0.0, mesh_max * 0.9)
		num_levels = max(self.contour_levels, 2)
		levels = np.linspace(mesh_min, mesh_max, num_levels)

		# Use matplotlib to extract contour paths
		import matplotlib
		matplotlib.use('Agg')
		from matplotlib.figure import Figure as MPLFigure
		fig = MPLFigure()
		ax = fig.add_subplot(111)
		# Create meshgrid for contour (centered on bin edges)
		xc = 0.5 * (xedges[:-1] + xedges[1:])
		yc = 0.5 * (yedges[:-1] + yedges[1:])
		CX, CY = np.meshgrid(xc, yc)
		try:
			cs = ax.contour(CX, CY, hist.T, levels=levels)
		except Exception:
			fig._destroy_aggs()
			if self._contour_group is not None:
				self._scene.remove(self._contour_group)
				self._contour_group = None
			return

		# Determine contour color from first visible dataset's pparm
		con_color = (0, 0, 0, 1.0)  # default black
		for key in self.data:
			if not self.visibility.get(key, True):
				continue
			pp = self.pparm.get(key)
			if pp is not None:
				con_color = _hex_to_rgba(COLORS[pp[0] % len(COLORS)])
				break

		# Extract contour segments and create pygfx Line objects
		con_group = gfx.Group()
		con_group.name = "contours"
		con_group.render_order = -1  # behind data points
		allsegs = cs.allsegs
		for level_segs in allsegs:
			for curve in level_segs:
				if len(curve) < 2:
					continue
				positions = np.column_stack([
					curve[:, 0].astype(np.float32),
					curve[:, 1].astype(np.float32),
					np.zeros(len(curve), dtype=np.float32)])
				mat = gfx.LineMaterial(thickness=1.0, color=con_color)
				con_group.add(gfx.Line(gfx.Geometry(positions=positions), mat))

		# Replace or add contour group in scene (before data groups)
		if self._contour_group is not None:
			self._scene.remove(self._contour_group)
		self._contour_group = con_group
		self._scene.add(con_group)

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

		# Read actual visible world bounds from camera frustum
		try:
			f = camera.frustum
			near = f[0]
			xleft, xright = float(near[:, 0].min()), float(near[:, 0].max())
			ybottom, ytop = float(near[:, 1].min()), float(near[:, 1].max())
		except Exception:
			xleft, xright = 0, 1
			ybottom, ytop = 0, 1
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

		# Compute display labels with log suffix if active
		x_lbl = self.xaxis_label
		if self.xlog and x_lbl:
			x_lbl = x_lbl + " (log10)"
		elif not x_lbl and self.xlog:
			x_lbl = "X (log10)"
		y_lbl = self.yaxis_label
		if self.ylog and y_lbl:
			y_lbl = y_lbl + " (log10)"
		elif not y_lbl and self.ylog:
			y_lbl = "Y (log10)"

		label_color = (0, 0, 0, 1.0)

		# Update axis labels in-place; create only on first pass
		if len(self._axis_labels) < 2:
			for lbl in self._axis_labels:
				try:
					self._scene.remove(lbl)
				except Exception:
					pass
			self._axis_labels = []

			if x_lbl:
				xlbl = gfx.Text(x_lbl, font_size=18, screen_space=True)
				xlbl.material.color = label_color
				sx_lbl = margin_left + (w - margin_left - margin_right) / 2
				sy_lbl = h - margin_bottom + 25
				xlbl._screen_pos = (sx_lbl, sy_lbl)
				xlbl.local.position = (*self._screen_to_world(sx_lbl, sy_lbl), 0)
				self._scene.add(xlbl)
				self._axis_labels.append(xlbl)

			if y_lbl:
				ylbl = gfx.Text(y_lbl, font_size=18, screen_space=True)
				ylbl.material.color = label_color
				sy_lbl = margin_top + (h - margin_top - margin_bottom) / 2 - 10
				ylbl._screen_pos = (10, sy_lbl)
				ylbl.local.position = (*self._screen_to_world(10, sy_lbl), 0)
				ylbl.local.rotation = la.quat_from_euler((0, 0, np.radians(90)))
				self._scene.add(ylbl)
				self._axis_labels.append(ylbl)
		else:
			# Update existing labels in-place (no scene graph changes)
			sx_lbl = margin_left + (w - margin_left - margin_right) / 2
			sy_lbl = h - margin_bottom + 25
			if x_lbl and len(self._axis_labels) > 0:
				xlbl = self._axis_labels[0]
				xlbl.text = x_lbl
				xlbl._screen_pos = (sx_lbl, sy_lbl)
				xlbl.local.position = (*self._screen_to_world(sx_lbl, sy_lbl), 0)
			if y_lbl and len(self._axis_labels) > 1:
				ylbl = self._axis_labels[1]
				ylbl.text = y_lbl
				sy_lbl = margin_top + (h - margin_top - margin_bottom) / 2 - 10
				ylbl._screen_pos = (10, sy_lbl)
				ylbl.local.position = (*self._screen_to_world(10, sy_lbl), 0)

		# Per-frame repositioning for zoom/pan changes
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
				self.axes[key] = (-1, 0)
			elif n_cols == 2:
				self.axes[key] = (0, 1)
			else:
				self.axes[key] = (0, 1)

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
				xa, ya = self.axes[key]
				if xa >= 0 and ya >= 0:
					self.set_axis_parms(
						str(column_labels[xa]), str(column_labels[ya]))
			except Exception:
				pass

		self.autoscale()

		# Set default contour bins based on number of data points (first load only)
		if is_new_key:
			ry = data_list[self.axes[key][1]]
			self.contour_bins = max(10, round(sqrt(len(ry)) / 5))

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

			x_idx, y_idx = ax_cfg
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

		# Apply log transforms to match what _rebuild_data_groups renders
		if self.xlog and np.min(x_finite) > 0:
			x_finite = np.log10(x_finite)
		if self.ylog and np.min(y_finite) > 0:
			y_finite = np.log10(y_finite)

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

	def _get_shape_group(self):
		"""Ensure the shape group exists."""
		if self._shape_group is None:
			self._shape_group = gfx.Group()
			self._shape_group.render_order = 50
			self._scene.add(self._shape_group)
		return self._shape_group

	def _get_or_create_node(self, name, node):
		"""Replace a named child in shape_group if it exists; else add."""
		group = self._get_shape_group()
		for child in list(group.children):
			if getattr(child, "name", None) == name:
				group.remove(child)
		node.name = name
		group.add(node)

	def _update_crosshairs(self, sx, sy, world_x, world_y, w, h):
		"""Create or update crosshair lines and coord label."""
		margin_left, margin_right, margin_top = 65, 20, 20
		color = (0.0, 0.0, 0.0)

		# Horizontal line
		hx1, hy1 = margin_left, sy
		hx2, hy2 = w - margin_right, sy
		wx_hy1 = self._screen_to_world(hx1, hy1)
		wx_hy2 = self._screen_to_world(hx2, hy2)
		verts = np.array([[wx_hy1[0], wx_hy1[1], 0.1], [wx_hy2[0], wx_hy2[1], 0.1]], dtype=np.float32)
		xline = gfx.Line(gfx.Geometry(positions=verts),
			gfx.LineMaterial(thickness=1.0, color=(color[0], color[1], color[2], 1.0)))
		xline.render_order = 999
		self._get_or_create_node("xcross", xline)

		# Vertical line
		vx1, vy1 = sx, 0
		vx2, vy2 = sx, h
		wx_vy1 = self._screen_to_world(vx1, vy1)
		wx_vy2 = self._screen_to_world(vx2, vy2)
		verts = np.array([[wx_vy1[0], wx_vy1[1], 0.1], [wx_vy2[0], wx_vy2[1], 0.1]], dtype=np.float32)
		yline = gfx.Line(gfx.Geometry(positions=verts),
			gfx.LineMaterial(thickness=1.0, color=(color[0], color[1], color[2], 1.0)))
		yline.render_order = 999
		self._get_or_create_node("ycross", yline)

		# Coord label (upper right inside plot area)
		label_text = "(%g, %g)" % (world_x, world_y)
		sx_lbl = w - margin_right - 100
		sy_lbl = margin_top + 30
		wx_lbl = self._screen_to_world(sx_lbl, sy_lbl)
		text_node = gfx.Text(label_text, material=gfx.TextMaterial(color='#000', outline_color='#fff', outline_thickness=0.1), font_size=16,
			anchor='middle-center', screen_space=True, render_order=999)
		text_node.local.position = (wx_lbl[0], wx_lbl[1], 0)
		text_node.material.color = (color[0], color[1], color[2], 1.0)
		self._get_or_create_node("coord", text_node)

	def _update_zoom_box(self, sx0, sy0, sx1, sy1):
		"""Create or update the rubber-band zoom rectangle."""
		color = (0.0, 0.0, 0.0)
		p0 = self._screen_to_world(sx0, sy0)
		p1 = self._screen_to_world(sx1, sy1)
		verts = np.array([
			[p0[0], p0[1], 0.1], [p1[0], p0[1], 0.1],
			[p1[0], p1[1], 0.1], [p0[0], p1[1], 0.1],
			[p0[0], p0[1], 0.1]], dtype=np.float32)
		rect = gfx.Line(gfx.Geometry(positions=verts),
			gfx.LineMaterial(thickness=1.5, color=(color[0], color[1], color[2], 1.0)))
		rect.render_order = 999
		self._get_or_create_node("zoombox", rect)

	def _clear_shapes(self):
		"""Remove all overlay annotations from the shape group."""
		if self._shape_group:
			while len(self._shape_group.children) > 0:
				self._shape_group.remove(self._shape_group.children[0])

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
			self._clear_shapes()
			self._right_drag_pos = (sx, sy)
			self._redraw = True
			self._request_render()
			return

		# Left click -> crosshair probe mode
		if event.button() == Qt.MouseButton.LeftButton:
			self._left_drag_active = True
			self._dirty = True
			self._update_crosshairs(sx, sy, world_pos[0], world_pos[1], w, h)
			self._request_render()

	def _on_mouse_move(self, event):
		sx, sy = event.position().x(), event.position().y()
		w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
		world_pos = self._screen_to_world(sx, sy)

		if self.mouseemit and self._left_drag_active:
			self.mousedrag.emit(event, world_pos)

		# Right-drag -> rubber-band zoom box
		if self._right_drag_pos is not None:
			x1r, y1r = self._right_drag_pos
			self._dirty = True
			self._update_zoom_box(x1r, y1r, sx, sy)
			self._request_render()
			return

		# Left-drag -> update crosshairs + coord label
		if self._left_drag_active:
			self._dirty = True
			self._update_crosshairs(sx, sy, world_pos[0], world_pos[1], w, h)
			self._request_render()

	def _on_mouse_release(self, event):
		sx, sy = event.position().x(), event.position().y()

		# Right release -> apply rubber-band zoom or rescale
		if self._right_drag_pos is not None:
			self._clear_shapes()
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
			self._clear_shapes()
			self._redraw = True
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

	def resizeEvent(self, event):
		super().resizeEvent(event)
		# After resize, re-apply camera limits so margins are correct
		if self.xlimits and self.ylimits:
			self._update_camera_limits()
			self._redraw = True
			self._request_render()



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

		# Flag to suppress _on_selection_changed during programmatic list rebuilds
		self._selection_suppressed = False

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

		# Slider for stepping through data sets
		self.showslide = ValSlider(label="Sel:", value=0)
		self.showslide.setIntonly(True)
		self.showslide.setRange(0, 30)
		self.showslide.setToolTip("Show only the data set at this index in the list")
		dsl.addWidget(self.showslide)

		# ns and stp boxes for slider range selection
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

		# ── Save / Analysis buttons ──
		btn_row = QtWidgets.QHBoxLayout()
		self.save_btn = QtWidgets.QPushButton("Save")
		self.save_btn.clicked.connect(self._on_save_plot)
		self.save_btn.setToolTip("Save plot to file (not yet implemented)")
		btn_row.addWidget(self.save_btn)
		self.stats_btn = QtWidgets.QPushButton("Statistics")
		self.stats_btn.clicked.connect(self._on_statistics)
		self.stats_btn.setToolTip("Print min/max/mean/std for each column of selected data")
		btn_row.addWidget(self.stats_btn)
		self.regress_btn = QtWidgets.QPushButton("Regression")
		self.regress_btn.clicked.connect(self._on_regression)
		self.regress_btn.setToolTip("Fit linear regression (y=mx+b) to selected columns, add as new set")
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

		# 3x3 grid: toggles on top row, then Line/Width/Bins, then Symbol/Size/Levels
		ag = QtWidgets.QGridLayout()

		self.line_tog = QtWidgets.QPushButton("Line")
		self.line_tog.setCheckable(True)
		self.line_tog.clicked.connect(self._on_appearance_change)
		self.line_tog.setToolTip("Toggle line display for selected data sets")
		self.sym_tog = QtWidgets.QPushButton("Symbol")
		self.sym_tog.setCheckable(True)
		self.sym_tog.clicked.connect(self._on_appearance_change)
		self.sym_tog.setToolTip("Toggle symbol markers for selected data sets")
		self.con_tog = QtWidgets.QPushButton("Contour")
		self.con_tog.setCheckable(True)
		self.con_tog.clicked.connect(self._on_contour_change)
		self.con_tog.setToolTip("Enable/disable density contour overlay")
		ag.addWidget(self.line_tog, 0, 0)
		ag.addWidget(self.sym_tog, 0, 1)
		ag.addWidget(self.con_tog, 0, 2)

		self.linetype_combo = QtWidgets.QComboBox()
		self.linetype_combo.addItems(["Solid", "Dashed", "Dotted", "Dash-Dot"])
		self.linetype_combo.setToolTip("Set line style for selected data sets")
		self.linewidth_spin = QtWidgets.QSpinBox()
		self.linewidth_spin.setRange(1, 10)
		self.linewidth_spin.valueChanged.connect(self._on_appearance_change)
		self.linewidth_spin.setToolTip("Set line thickness (1-10 pixels)")
		self.contour_bins_spin = QtWidgets.QSpinBox()
		self.contour_bins_spin.setRange(10, 500)
		self.contour_bins_spin.setValue(100)
		self.contour_bins_spin.valueChanged.connect(self._on_contour_change)
		self.contour_bins_spin.setToolTip("Grid resolution for contour density estimation")
		ag.addWidget(self.linetype_combo, 1, 0)
		ag.addWidget(self.linewidth_spin, 1, 1)
		ag.addWidget(self.contour_bins_spin, 1, 2)

		self.symtype_combo = QtWidgets.QComboBox()
		self.symtype_combo.addItems(["Circle", "Square", "Plus", "TriUp", "TriDown"])
		self.symtype_combo.setToolTip("Set marker shape for selected data sets")
		self.symsize_spin = QtWidgets.QSpinBox()
		self.symsize_spin.setRange(1, 30)
		self.symsize_spin.valueChanged.connect(self._on_appearance_change)
		self.symsize_spin.setToolTip("Set marker size (1-30 pixels)")
		self.contour_levels_spin = QtWidgets.QSpinBox()
		self.contour_levels_spin.setRange(2, 50)
		self.contour_levels_spin.setValue(15)
		self.contour_levels_spin.valueChanged.connect(self._on_contour_change)
		self.contour_levels_spin.setToolTip("Number of contour lines to draw")
		ag.addWidget(self.symtype_combo, 2, 0)
		ag.addWidget(self.symsize_spin, 2, 1)
		ag.addWidget(self.contour_levels_spin, 2, 2)

		apl.addLayout(ag)

		vbl.addWidget(apg)

		# ── Column selectors ──
		cg = QtWidgets.QGroupBox("Columns")
		cl = QtWidgets.QGridLayout()
		xlabel = QtWidgets.QLabel("X:")
		xlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
		cl.addWidget(xlabel, 0, 0)
		self.col_x_spin = QtWidgets.QSpinBox()
		self.col_x_spin.setRange(-1, 5)
		self.col_x_spin.setToolTip("Column index for X axis (-1 = row index)")
		cl.addWidget(self.col_x_spin, 0, 1)
		ylabel = QtWidgets.QLabel("Y:")
		ylabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
		cl.addWidget(ylabel, 0, 2)
		self.col_y_spin = QtWidgets.QSpinBox()
		self.col_y_spin.setRange(-1, 5)
		self.col_y_spin.setToolTip("Column index for Y axis (-1 = row index)")
		cl.addWidget(self.col_y_spin, 0, 3)
		cg.setLayout(cl)
		vbl.addWidget(cg)

		# ── Axis scale toggles ──
		scale_row = QtWidgets.QHBoxLayout()
		self.xlog_tog = QtWidgets.QPushButton("X Log")
		self.xlog_tog.setCheckable(True)
		self.xlog_tog.setToolTip("Toggle log10 scale on X axis")
		scale_row.addWidget(self.xlog_tog)
		self.ylog_tog = QtWidgets.QPushButton("Y Log")
		self.ylog_tog.setCheckable(True)
		self.ylog_tog.setToolTip("Toggle log10 scale on Y axis")
		scale_row.addWidget(self.ylog_tog)
		self.rescale_btn = QtWidgets.QPushButton("Rescale")
		self.rescale_btn.setToolTip("Auto-scale axes to fit all visible data")
		scale_row.addWidget(self.rescale_btn)
		vbl.addLayout(scale_row)

		# ── Axis limits ──
		limit_g = QtWidgets.QGridLayout()
		self.xmin_box = ValBox(label="X:", value=0)
		self.xmin_box.setToolTip("X axis minimum")
		self.xmax_box = ValBox(label="", value=1)
		self.xmax_box.setToolTip("X axis maximum")
		self.ymin_box = ValBox(label="Y:", value=0)
		self.ymin_box.setToolTip("Y axis minimum")
		self.ymax_box = ValBox(label="", value=1)
		self.ymax_box.setToolTip("Y axis maximum")
		limit_g.addWidget(self.xmin_box, 0, 0)
		limit_g.addWidget(self.xmax_box, 0, 1)
		limit_g.addWidget(self.ymin_box, 1, 0)
		limit_g.addWidget(self.ymax_box, 1, 1)
		vbl.addLayout(limit_g)

		# ── Labels only (no title) ──
		label_g = QtWidgets.QGridLayout()
		self.xlabel_edit = QtWidgets.QLineEdit()
		self.xlabel_edit.setToolTip("X axis title text")
		self.ylabel_edit = QtWidgets.QLineEdit()
		self.ylabel_edit.setToolTip("Y axis title text")
		label_g.addWidget(QtWidgets.QLabel("X Label:"), 0, 0)
		label_g.addWidget(self.xlabel_edit, 0, 1)
		label_g.addWidget(QtWidgets.QLabel("Y Label:"), 1, 0)
		label_g.addWidget(self.ylabel_edit, 1, 1)
		vbl.addLayout(label_g)

		# ── Transparency slider ──
		self.alpha_slider = ValSlider(label="Alpha", value=0.8)
		self.alpha_slider.setRange(0.1, 1.0)
		self.alpha_slider.setToolTip("Transparency for selected data sets (0.1-1.0)")
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

		# Item checkbox changed -> visibility toggle
		self.setlist.itemChanged.connect(self._on_item_changed)

		# Slider for stepping through data sets
		self.showslide.valueChanged.connect(self._on_slide_change)

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

		# Contour settings
		self.con_tog.setChecked(tgt.contour_enabled)
		self.contour_bins_spin.setValue(tgt.contour_bins)
		self.contour_levels_spin.setValue(tgt.contour_levels)

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

		# Column selectors - disconnect signals to avoid spurious updates
		try:
			self.col_x_spin.valueChanged.disconnect(self._on_column_change)
		except Exception:
			pass
		try:
			self.col_y_spin.valueChanged.disconnect(self._on_column_change)
		except Exception:
			pass

		# Compute dynamic column range from data
		max_cols = 0
		for k, dl in tgt.data.items():
			if isinstance(dl, list) and len(dl) > max_cols:
				max_cols = len(dl)
		self.col_x_spin.setRange(-1, max(5, max_cols - 1))
		self.col_y_spin.setRange(-1, max(5, max_cols - 1))

		ax_cfg = tgt.axes.get(key, (-1, 0))
		self.col_x_spin.setValue(ax_cfg[0])
		self.col_y_spin.setValue(ax_cfg[1])

		# Reconnect signals
		self.col_x_spin.valueChanged.connect(self._on_column_change)
		self.col_y_spin.valueChanged.connect(self._on_column_change)

	def _on_selection_changed(self, row):
		"""Called when the user selects a data set in the list.

		Always syncs inspector controls from the first/primary selected item,
		even during multi-select. This gives a guaranteed valid baseline;
		individual property changes then apply to all currently selected items.
		Not suppressed during programmatic list rebuilds.
		"""
		if self._selection_suppressed:
			return
		item = self.setlist.selectedItems()[0] if self.setlist.selectedItems() else None
		if item is None:
			return
		key = item.data(QtCore.Qt.UserRole)
		self._update_controls_for_key(key)

	def _selected_keys(self):
		"""Get keys of all selected data sets (supports multi-select)."""
		keys = []
		for item in self.setlist.selectedItems():
			key = item.data(QtCore.Qt.UserRole)
			if key is not None:
				keys.append(key)
		return keys

	def _selected_key(self):
		"""Get the primary selected data set key (first selection)."""
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
		"""Handle checkbox state changes for visibility toggling."""
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
		"""Handle slider stepping for selective visibility."""
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
		"""Rebuild the data set list, preserving selection."""
		tgt = self._get_tgt()
		if not tgt:
			return

		# Remember current selection row (index)
		old_row = self.setlist.currentRow()

		# Block itemChanged signal during list rebuild
		self.setlist.blockSignals(True)
		try:
			self.setlist.clear()

			for key in tgt.data:
				pp = tgt.pparm.get(key)
				color_idx = pp[0] % len(COLORS) if pp else 0

				item = QtWidgets.QListWidgetItem(key)
				item.setData(QtCore.Qt.UserRole, key)

				# Set checkbox flag
				flags = item.flags()
				flags |= QtCore.Qt.ItemIsUserCheckable
				item.setFlags(flags)

				# Set check state based on visibility
				if tgt.visibility.get(key, True):
					item.setCheckState(QtCore.Qt.Checked)
				else:
					item.setCheckState(QtCore.Qt.Unchecked)

				# Color the text to match the plot color
				if color_idx in COLOR_RGB:
					item.setForeground(COLOR_RGB[color_idx])

				self.setlist.addItem(item)
		finally:
			self.setlist.blockSignals(False)

		# Update slider range to match number of data sets
		try:
			self.showslide.valueChanged.disconnect(self._on_slide_change)
		except Exception:
			pass
		num_sets = self.setlist.count()
		if num_sets > 1:
			self.showslide.setRange(0, num_sets - 1)
		else:
			self.showslide.setRange(0, 0)
		self.showslide.valueChanged.connect(self._on_slide_change)

		# Restore selection (suppress handler to avoid unwanted sync)
		if 0 <= old_row < self.setlist.count():
			self._selection_suppressed = True
			try:
				self.setlist.setCurrentRow(old_row)
			finally:
				self._selection_suppressed = False

		# On initial load (nothing previously selected), select first item and sync controls
		if old_row < 0 and self.setlist.count() > 0:
			self._selection_suppressed = True
			try:
				self.setlist.setCurrentRow(0)
			finally:
				self._selection_suppressed = False
			# Sync inspector from the first selected item without suppression
			item = self.setlist.item(0)
			if item is not None:
				key = item.data(QtCore.Qt.UserRole)
				self._update_controls_for_key(key)

	def datachange(self):
		self.update_list()

	def _on_save_plot(self):
		tgt = self._get_tgt()
		if tgt:
			pass  # Implement later

	def _on_statistics(self):
		"""Compute and print summary statistics for each column of the selected data set."""
		key = self._selected_key()
		if key is None or not self.target():
			return
		tgt = self.target()

		dl = tgt.data.get(key)
		if not dl:
			return

		n_cols = len(dl)
		n_rows = max(len(c) for c in dl) if dl else 0
		if n_rows == 0:
			print(f"Statistics: no data in '{key}'")
			return

		print(f"\n=== Statistics for '{key}' ({n_rows} rows, {n_cols} columns) ===")
		print(f"{'Col':<6} {'Min':>14} {'Max':>14} {'Mean':>14} {'Std':>14}")
		print("-" * 62)

		for col_idx in range(n_cols):
			arr = np.asarray(dl[col_idx], dtype=np.float64)
			if len(arr) < n_rows:
				# Pad with NaN if column is shorter
				padded = np.full(n_rows, np.nan)
				padded[:len(arr)] = arr
				arr = padded

			cmin = float(np.min(arr))
			cmax = float(np.max(arr))
			cmean = float(np.mean(arr))
			cstd = float(np.std(arr, ddof=0))  # population std dev

			print(f"{col_idx:<6} {cmin:>14.6g} {cmax:>14.6g} {cmean:>14.6g} {cstd:>14.6g}")
		print()

	def _on_regression(self):
		"""Perform linear regression on the selected data set.

		Creates a new data set named "d_ycol = m d_xcol + b" with 5 points
		at x coordinates spanning min-max range with 5% padding.
		"""
		key = self._selected_key()
		if key is None or not self.target():
			return
		tgt = self.target()

		dl = tgt.data.get(key)
		if not dl:
			return

		x_idx, y_idx = tgt.axes.get(key, (0, 1))

		# Handle x_idx == -1 (index-based)
		if x_idx < 0:
			x_vals = np.arange(len(dl[y_idx]), dtype=np.float64)
		else:
			x_vals = np.asarray(dl[x_idx], dtype=np.float64)
		y_vals = np.asarray(dl[y_idx], dtype=np.float64)

		n = len(x_vals)
		if n < 2 or len(y_vals) != n:
			return

		# Simple linear regression: y = m*x + b
		x_mean = np.mean(x_vals)
		y_mean = np.mean(y_vals)
		sxy = np.sum((x_vals - x_mean) * (y_vals - y_mean))
		sxx = np.sum((x_vals - x_mean) ** 2)

		if sxx == 0:
			m, b = 0.0, y_mean
		else:
			m = sxy / sxx
			b = y_mean - m * x_mean

		x_min = float(np.min(x_vals))
		x_max = float(np.max(x_vals))
		w = x_max - x_min

		# 5 fit points: min-5%, min, mid, max, max+5%
		fit_x = np.array([
			x_min - w * 0.05,
			x_min,
			(x_min + x_max) / 2.0,
			x_max,
			x_max + w * 0.05,
		], dtype=np.float64)
		fit_y = m * fit_x + b

		# Build new dataset with same column count, zeros except x and y columns
		n_cols = len(dl)
		new_data = []
		for col in range(n_cols):
			if col == y_idx:
				new_data.append(fit_y.copy())
			elif x_idx >= 0 and col == x_idx:
				new_data.append(fit_x.copy())
			else:
				new_data.append(np.zeros(5, dtype=np.float64))

		# Generate name like "d_1 = 2.34 d_0 + -0.12"
		reg_key = f"d_{y_idx} = {m:.5g} d_{x_idx if x_idx >= 0 else '?'} + {b:.5g}"
		print(f"Regression: y = {m:.8g} * x + {b:.8g}")

		tgt.set_data(new_data, key=reg_key)
		tgt._dirty = True
		tgt._request_render()
		self.update_list()

	def _on_appearance_change(self):
		"""Handle individual appearance control changes for multi-select.

		Only applies the specific property that changed to all selected items.
		"""
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
			# Also trigger contour rebuild if enabled
			if tgt.contour_enabled:
				con_group = tgt._contour_group
				if con_group is not None:
					while len(con_group.children) > 0:
						con_group.remove(con_group.children[0])
			tgt._request_render()

	def _on_color_change(self, idx):
		keys = self._selected_keys()
		if not keys or not self.target():
			return
		tgt = self.target()

		for key in keys:
			pp = tgt.pparm.get(key)
			if pp is None:
				continue
			pp = list(pp)
			pp[0] = idx % len(COLORS)
			tgt.pparm[key] = pp

		# Update the list display to reflect new color
		self.update_list()
		tgt._dirty = True
		tgt._request_render()

	def _on_column_change(self):
		key = self._selected_key()
		if key is None or not self.target():
			return
		tgt = self.target()

		x_idx = self.col_x_spin.value()
		y_idx = self.col_y_spin.value()
		tgt.axes[key] = (x_idx,y_idx)

		tgt.autoscale()
		# Set dirty AFTER autoscale so limits are computed before rebuild starts
		tgt._dirty = True
		tgt._request_render()

	def _on_scale_change(self):
		tgt = self._get_tgt()
		if tgt:
			tgt.xlog = self.xlog_tog.isChecked()
			tgt.ylog = self.ylog_tog.isChecked()
			tgt._dirty = True
			tgt._request_render()

	def _on_contour_change(self):
		tgt = self._get_tgt()
		if tgt:
			tgt.contour_enabled = self.con_tog.isChecked()
			tgt.contour_bins = self.contour_bins_spin.value()
			tgt.contour_levels = self.contour_levels_spin.value()
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
					tgt._camera.show_rect(xmin, xmax, ymax, ymin, depth=20)
				except Exception:
					pass  # show_rect can fail with degenerate limits
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
		keys = self._selected_keys()
		if not keys or not self.target():
			return
		tgt = self.target()

		for key in keys:
			pp = tgt.pparm.get(key)
			if pp is None:
				continue
			pp = list(pp)
			while len(pp) <= 10:
				pp.append(0.8)
			pp[10] = val
			tgt.pparm[key] = pp

		tgt._dirty = True
		tgt._request_render()

	def closeEvent(self, event):
		tgt = self._get_tgt()
		if tgt:
			tgt.inspector = None
		super().closeEvent(event)
