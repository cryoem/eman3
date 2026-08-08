#!/usr/bin/env python
#
# EMAN3 Multi-Image Matrix Viewer - PyGfx rendering with PySide6
# Ported from EMAN2 qtgui/emimagemx.py


import weakref
import os
import math

import pygfx as gfx
from rendercanvas.qt import QRenderWidget

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import Qt
from EMAN3.gui.valslider import ValSlider

import numpy as np

# Import rendering helpers from emimage2d
from EMAN3.gui.emimage2d import (
	_render_image_8bit,
)


# Set outline colors (RGBA floats)
# SET_COLORS[0] (red) is reserved for the "Deleted" set.
# Other sets use indices 1..N, skipping index 0.
SET_COLORS = [
	(1.0, 0.0, 0.0, 1.0),       # red - reserved for Deleted set only
	(0.0, 0.0, 1.0, 1.0),       # blue
	(0.0, 0.85, 0.0, 1.0),      # green
	(0.0, 0.8, 0.8, 1.0),       # cyan
	(0.502, 0.0, 0.502, 1.0),   # purple
	(1.0, 0.647, 0.0, 1.0),     # orange
	(1.0, 1.0, 0.0, 1.0),       # yellow
	(1.0, 0.412, 0.706, 1.0),   # hotpink
	(0.831, 0.698, 0.0, 1.0),   # gold
]


class _EMCanvas(QRenderWidget):
	"""QRenderWidget subclass that forwards mouse events to the parent widget."""

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


class EMImageMXWidget(QtWidgets.QWidget):
	"""Tiled multi-image viewer using PyGfx for rendering.

	Displays a grid of same-sized 2D images with metadata labels,
	set membership tagging, and scrollable navigation.
	"""

	mx_boxdeleted = QtCore.Signal(object, list, bool)
	mx_image_selected = QtCore.Signal(object, tuple)
	mx_image_double = QtCore.Signal(object, tuple)
	mouseup = QtCore.Signal(object, tuple)
	keypress = QtCore.Signal(object)

	def __init__(self, data=None, parent=None):
		super().__init__(parent)

		self.setFocusPolicy(Qt.StrongFocus)
		self.setMouseTracking(True)
		self.setMinimumSize(256, 256)

		# Data state
		self._data = None             # list of numpy arrays (ny, nx) float32
		self._data_headers = []       # list of dicts with header metadata
		self._img_xsize = 0
		self._img_ysize = 0
		self.nimg = 0

		self.file_name = ""
		self.img_num_offset = 0

		# Display params (shared across all images)
		self.scale = 1.0
		self.minden = 0.0
		self.maxden = 1.0
		# Current contrast values (modified by Brt/Cont, bounded by minden/maxden)
		self.curmin = 0.0
		self.curmax = 1.0
		self.invert = False
		# Full data range across all images (set by set_data/auto_contrast)
		self._global_min = 0.0
		self._global_max = 1.0
		self.gamma = 1.0
		self.brightness = 0.0
		self.contrast = 1.0

		# Grid layout
		self.min_sep = 4             # min separation between tiles (px)
		self.scroll_offset = 0       # vertical scroll offset in px

		# Mouse mode
		self.mouse_modes = ["App", "Del", "Sets"]
		self.mmode = "App"
		# Right-drag scroll state
		self._right_drag_active = False
		self._right_drag_start = None

		# Sets system
		self.sets = {}              # name -> set of image indices
		self.sets_visible = {}      # name -> set of visible indices
		self.current_set = None
		self.selected = []

		# Deleted images
		self._deleted_idxs = set()

		# Metadata display
		self.valstodisp = ["Img #"]
		self.font_size = 12

		# PyGfx setup
		self._canvas = None
		self._renderer = None
		self._scene = gfx.Scene()
		self._camera = None
		self._dirty = True

		# Scene groups (rendered in order)
		self._image_group = gfx.Group()
		self._set_outline_group = gfx.Group()
		self._text_group = gfx.Group()
		self._shape_group = gfx.Group()

		self._scene.add(self._image_group)
		self._scene.add(self._set_outline_group)
		self._scene.add(self._text_group)
		self._scene.add(self._shape_group)

		self._setup_gfx()

		# Inspector
		self.inspector = None

		# Defer data loading
		if data is not None:
			QtCore.QTimer.singleShot(0, lambda: self.set_data(data))

	def _setup_gfx(self):
		"""Initialize PyGfx rendering pipeline."""
		try:
			self._canvas = _EMCanvas(parent_widget=self, parent=self)

			layout = QtWidgets.QVBoxLayout(self)
			layout.setContentsMargins(0, 0, 0, 0)
			layout.addWidget(self._canvas)

			self._renderer = gfx.WgpuRenderer(self._canvas)

			# Orthographic camera for screen-aligned tiled display
			self._camera = gfx.OrthographicCamera(maintain_aspect=False)
			self._camera.world.position = (0, 0, 1)

			# Add the background object to the scene
			bg_material = gfx.BackgroundMaterial((0.0, 0.0, 0.0, 1.0))
			background = gfx.Background(None, bg_material)
			self._scene.add(background)

			# Initialize text material with a dummy node
			try:
				_dummy_text = gfx.Text("Init", font_size=12, render_order=999)
				_dummy_text.material.color = (0, 0, 0, 0)
				_dummy_text.local.position = (0, 0, 2)
				self._scene.add(_dummy_text)
			except Exception:
				pass

			self._canvas.request_draw(self._render_callback)
			self.setAcceptDrops(True)
			self.setContextMenuPolicy(Qt.PreventContextMenu)
		except Exception as e:
			print(f"Warning: PyGfx init failed ({e})")
			self._renderer = None
			self._camera = None

	# ─── Rendering ──────────────────────────────────────────────

	def _render_callback(self):
		"""Called by canvas to render a frame."""
		if self._renderer is None or self._camera is None:
			return

		w = self.width()
		h = self.height()
		if w == 0 or h == 0:
			self._canvas.request_draw(self._render_callback)
			return

		self._camera.width = w
		self._camera.height = h
		self._camera.world.position = (w / 2, h / 2, 1)

		if self._dirty:
			self._dirty = False
			self._render()

		try:
			self._renderer.render(self._scene, self._camera)
		except Exception as e:
			import traceback
			print(f"Render error: {e}")
			traceback.print_exc()

		self._canvas.request_draw(self._render_callback)

	def _request_render(self):
		"""Mark dirty and request a re-render."""
		self._dirty = True
		if self._canvas:
			self._canvas.request_draw(self._render_callback)

	def _compute_grid(self):
		"""Compute which images are visible given current scale/scroll.

		Returns (rowstart, visiblerows, visiblecols, total_rows).
		"""
		if self._data is None or self.nimg == 0:
			return (0, 0, 0, 0)

		w = self.width()
		h = self.height()
		if w <= 0 or h <= 0:
			return (0, 0, 0, 0)

		rendered_w = self._img_xsize * self.scale
		rendered_h = self._img_ysize * self.scale

		if rendered_w <= 0:
			return (0, 0, 0, 0)

		visiblecols = max(1, int(math.floor(w / (rendered_w + self.min_sep))))
		total_rows = math.ceil(self.nimg / visiblecols)

		row_h = rendered_h + self.min_sep

		if self.scroll_offset < 0:
			ybelow = math.floor(-self.scroll_offset // row_h)
			rowstart = ybelow
		else:
			rowstart = int(self.scroll_offset // row_h) if row_h > 0 else 0

		if rowstart < 0:
			rowstart = 0

		visiblerows = max(1, int(math.ceil(h / row_h)) + 1)
		if rowstart + visiblerows > total_rows:
			visiblerows = max(1, total_rows - rowstart)

		return (rowstart, visiblerows, visiblecols, total_rows)

	def _render(self):
		"""Render the full grid of images."""
		if self._data is None or self.nimg == 0:
			return

		self._clear_scene_groups()

		rowstart, visiblerows, visiblecols, total_rows = self._compute_grid()
		rendered_w = self._img_xsize * self.scale
		rendered_h = self._img_ysize * self.scale

		for row in range(rowstart, rowstart + visiblerows):
			for col in range(visiblecols):
				idx = row * visiblecols + col
				if idx >= self.nimg:
					break
				if idx in self._deleted_idxs:
					continue

				self._render_tile(idx, row, col, rendered_w, rendered_h, visiblecols)

		self._render_set_outlines(rowstart, visiblerows, visiblecols, rendered_w, rendered_h)
		self._render_labels(rowstart, visiblerows, visiblecols, rendered_w, rendered_h)

	def _clear_scene_groups(self):
		"""Remove all children from scene groups."""
		for group in (self._image_group, self._set_outline_group,
		              self._text_group, self._shape_group):
			while group.children:
				group.remove(group.children[0])

	def _render_tile(self, idx, row, col, rw, rh, visiblecols):
		"""Render a single image tile as a textured quad."""
		data = self._data[idx]
		if data is None:
			return

		w = self.width()
		h = self.height()

		tx = col * (rw + self.min_sep) + self.min_sep
		ty = row * (rh + self.min_sep) + self.min_sep - self.scroll_offset

		tw = min(rw, w - tx)
		th = min(rh, h - ty)

		if tw <= 0 or th <= 0:
			return

		try:
			int_tw = max(1, int(tw))
			int_th = max(1, int(th))

			rgb, _ = _render_image_8bit(
				data, self.scale, 0, 0,
				int_tw, int_th,
				self.curmin, self.curmax,
				gamma=self.gamma,
				invert=self.invert,
			)

			if rgb is None:
				return

			# Convert RGB to RGBA for texture
			rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.float32)
			rgba[:, :, :3] = rgb
			rgba[:, :, 3] = 1.0

			texture = gfx.Texture(rgba, dim=2)
			geometry = gfx.Geometry(grid=texture)
			material = gfx.ImageBasicMaterial()
			image_node = gfx.Image(geometry=geometry, material=material)

			image_node.world.position = (tx,ty, 0)	# Position needs to be lower left corner of image
			image_node.world_bounds = ((tx, ty, 0), (tx + tw, ty + th, 0))

			self._image_group.add(image_node)

		except Exception as e:
			print(f"Error rendering tile {idx}: {e}")

	def _render_set_outlines(self, rowstart, visiblerows, visiblecols, rw, rh):
		"""Draw colored outlines for images belonging to visible sets."""
		if not self.sets_visible:
			return

		sorted_sets = sorted(self.sets.items())
		n = self.nimg

		for row in range(rowstart, rowstart + visiblerows):
			for col in range(visiblecols):
				idx = row * visiblecols + col
				if idx >= n or idx in self._deleted_idxs:
					continue

				tx = col * (rw + self.min_sep) + self.min_sep
				ty = row * (rh + self.min_sep) + self.min_sep - self.scroll_offset

				for set_i, (set_name, set_list) in enumerate(sorted_sets):
					if set_name not in self.sets_visible:
						continue
					if idx not in set_list:
						continue

					color = SET_COLORS[set_i % len(SET_COLORS)]
					self._draw_set_outline(tx, ty, rw, rh, color)

	def _draw_set_outline(self, tx, ty, rw, rh, color):
		"""Draw a small colored square outline for set membership."""
		outline_w = 8
		outline_h = 8

		ox = tx + rw - outline_w - self.min_sep
		oy = ty + rh - outline_h - self.min_sep

		hw, hh = outline_w / 2, outline_h / 2
		cx, cy = ox + outline_w / 2, oy + outline_h / 2

		vertices = np.array([
			[cx - hw, cy - hh, 0.5],
			[cx + hw, cy - hh, 0.5],
			[cx + hw, cy + hh, 0.5],
			[cx - hw, cy + hh, 0.5],
			[cx - hw, cy - hh, 0.5],
		], dtype=np.float32)

		geometry = gfx.Geometry(positions=vertices,indices=[[0,1,2],[0,2,3]])
		# material = gfx.LineMaterial(thickness=2.0, color=color)
		material = gfx.MeshBasicMaterial(color=color)
		square = gfx.Mesh(geometry, material)
		self._set_outline_group.add(square)

	def _render_labels(self, rowstart, visiblerows, visiblecols, rw, rh):
		"""Render metadata text labels for each visible tile."""
		n = self.nimg

		for row in range(rowstart, rowstart + visiblerows):
			for col in range(visiblecols):
				idx = row * visiblecols + col
				if idx >= n or idx in self._deleted_idxs:
					continue

				tx = col * (rw + self.min_sep) + self.min_sep
				ty = row * (rh + self.min_sep) + self.min_sep - self.scroll_offset

				label_text = self._get_label_text(idx)
				if not label_text:
					continue

				text_node = gfx.Text(label_text, font_size=self.font_size, render_order=999, anchor="bottom-left", material=gfx.TextMaterial(color='#fff',outline_color='#000', outline_thickness=0.5))
				text_node.material.color = (1.0, 1.0, 1.0, 1.0)
				text_node.local.position = (tx+2, ty+2, 0.5)

				self._text_group.add(text_node)

	def _get_label_text(self, idx):
		"""Get the metadata label string for an image."""
		parts = []
		for val_name in self.valstodisp:
			if val_name == "Img #":
				parts.append(str(idx + self.img_num_offset))
			elif self._data_headers and idx < len(self._data_headers):
				header = self._data_headers[idx]
				if isinstance(header, dict):
					val = header.get(val_name, "?")
					parts.append(str(val))
		return "\n".join(parts)

	# ─── Data handling ──────────────────────────────────────────

	def set_data(self, obj, filename="", metadata=None):
		"""Set the data to display.

		Accepts a list of numpy 2D arrays (float), or an iterable of images.
		"""
		self.file_name = filename
		if filename:
			self.setWindowTitle(os.path.basename(filename))

		images = []
		headers = []

		if isinstance(obj, (list, tuple)):
			for item in obj:
				if isinstance(item, np.ndarray):
					arr = item.astype(np.float32)
					if arr.ndim == 2:
						images.append(arr)
					elif arr.ndim == 3 and arr.shape[0] == 1:
						images.append(arr[0])
					else:
						images.append(None)
					headers.append({})
				else:
					images.append(None)
					headers.append({})
		else:
			arr = np.asarray(obj, dtype=np.float32)
			if arr.ndim == 2:
				images = [arr]
				headers = [metadata or {}]

		self._data = images
		self._data_headers = headers
		self.nimg = len(images)

		# Get image dimensions from first valid image
		for img in images:
			if img is not None and img.ndim == 2:
				self._img_ysize, self._img_xsize = img.shape
				break

		# Auto-contrast on new data
		self.auto_contrast(inspector_update=False)

		self._dirty = True
		self._request_render()

	def clear_data(self):
		"""Clear all displayed data."""
		self._data = None
		self._data_headers = []
		self.nimg = 0
		self._img_xsize = 0
		self._img_ysize = 0
		self.sets = {}
		self.sets_visible = {}
		self._deleted_idxs.clear()

		self._clear_scene_groups()
		self._dirty = True
		self._request_render()

	# ─── Display parameter setters ──────────────────────────────

	def set_scale(self, val):
		self.scale = max(0.01, val)
		self.scroll_offset = 0
		self._dirty = True
		self._request_render()

	def get_scale(self):
		return self.scale

	def set_density_min(self, val):
		self.curmin = float(val)
		self._dirty = True
		self._request_render()

	def get_density_min(self):
		return self.curmin

	def set_density_max(self, val):
		self.curmax = float(val)
		self._dirty = True
		self._request_render()

	def get_density_max(self):
		return self.curmax

	def set_den_range(self, mn, mx):
		self.curmin = float(mn)
		self.curmax = float(mx)
		self._dirty = True
		self._request_render()

	def set_gamma(self, val):
		self.gamma = max(0.1, val)
		self._dirty = True
		self._request_render()

	def get_gamma(self):
		return self.gamma

	def set_invert(self, val):
		self.invert = bool(val)
		self._dirty = True
		self._request_render()

	def set_mouse_mode(self, mode):
		if mode in self.mouse_modes:
			self.mmode = mode

	def get_mmode(self):
		return self.mmode

	def set_display_values(self, vals):
		self.valstodisp = list(vals)
		self._dirty = True
		self._request_render()

	def set_file_name(self, name):
		self.file_name = name
		self.setWindowTitle(name if name else "")

	def get_image(self, idx):
		"""Get the raw numpy array for image at index."""
		if self._data and 0 <= idx < len(self._data):
			return self._data[idx]
		return None

	def set_font_size(self, val):
		self.font_size = max(6, min(48, int(val)))
		self._dirty = True
		self._request_render()

	def get_font_size(self):
		return self.font_size

	def auto_contrast(self, inspector_update=True):
		"""Auto-contrast: mean +/- 3 sigma sampled across images."""
		if not self._data or self.nimg == 0:
			return

		valid = [img for img in self._data if img is not None]
		if not valid:
			return

		n_sample = min(len(valid), 64)
		stp = max(len(valid) // n_sample, 1)

		mean_sum = 0.0
		max_sigma = 0.0
		global_min = float('inf')
		global_max = float('-inf')
		count = 0

		for i in range(0, len(valid), stp):
			img = valid[i]
			if img is None:
				continue
			mean_sum += float(np.mean(img))
			count += 1
			max_sigma = max(max_sigma, float(np.std(img)))
			global_min = min(global_min, float(np.min(img)))
			global_max = max(global_max, float(np.max(img)))

		if count == 0:
			return

		mean_sum /= count
		
		self._global_min = global_min
		self._global_max = global_max
		
		# minden/maxden store the full data range
		self.minden = global_min
		self.maxden = global_max
		
		# curmin/curmax are the actual contrast values (mean±3σ)
		self.curmin = max(global_min, mean_sum - 3.0 * max_sigma)
		self.curmax = min(global_max, mean_sum + 3.0 * max_sigma)

		self._clear_scene_groups()
		self._dirty = True
		self._request_render()

		if inspector_update and self.inspector:
			self.inspector._sync_from_widget()

	def full_contrast(self, inspector_update=True):
		"""Full contrast: use entire data range for contrast."""
		if self._global_max == self._global_min:
			return
		
		# Reset contrast to full range
		self.curmin = self.minden  # which equals _global_min
		self.curmax = self.maxden  # which equals _global_max

		self._clear_scene_groups()
		self._dirty = True
		self._request_render()

		if self.inspector:
			self.inspector._sync_from_widget()

	# ─── Sets management ────────────────────────────────────────

	def enable_set(self, name, lst=None, display=True, force=False):
		"""Create or update a set."""
		if name not in self.sets or force:
			self.sets[name] = set(lst) if lst else set()
		if display and name in self.sets:
			self.sets_visible[name] = self.sets[name]
		if self.current_set is None:
			self.current_set = name
		self._dirty = True
		self._request_render()

	def show_set(self, name):
		if name not in self.sets:
			return
		self.sets_visible[name] = self.sets[name]
		self._dirty = True
		self._request_render()

	def hide_set(self, name):
		self.sets_visible.pop(name, None)
		self._dirty = True
		self._request_render()

	def delete_set(self, names):
		if isinstance(names, str):
			names = [names]
		for n in names:
			self.sets.pop(n, None)
			self.sets_visible.pop(n, None)
		self._dirty = True
		self._request_render()

	def get_set(self, name):
		if name not in self.sets:
			self.sets[name] = set()
		return self.sets[name]

	def clear_sets(self):
		self.sets_visible.clear()
		self._dirty = True
		self._request_render()

	# ─── Scroll handling ────────────────────────────────────────

	def _compute_scroll_range(self):
		"""Compute the min/max scroll offset."""
		if self.nimg == 0 or not self._data:
			return (0, 0)

		rendered_h = self._img_ysize * self.scale
		row_h = rendered_h + self.min_sep
		rw = max(self._img_xsize * self.scale, 1)
		visiblecols = max(1, int(math.floor(self.width() / (rw + self.min_sep))))
		total_rows = math.ceil(self.nimg / visiblecols)
		total_height = total_rows * row_h

		return (0, max(0, total_height - self.height()))

	def scroll_by(self, delta):
		self.scroll_offset += delta
		min_sc, max_sc = self._compute_scroll_range()
		self.scroll_offset = max(min_sc, min(max_sc, self.scroll_offset))
		self._dirty = True
		self._request_render()

	# ─── Inspector ──────────────────────────────────────────────

	def get_inspector(self):
		if self.inspector is None:
			self.inspector = EMImageInspectorMX(self)
		return self.inspector

	def show_inspector(self):
		insp = self.get_inspector()
		if not insp.isVisible():
			insp.show()

	def inspector_update(self):
		self._dirty = True
		self._request_render()

	def force_display_update(self):
		self._dirty = True
		self._request_render()

	# ─── Mouse handling ─────────────────────────────────────────

	def _on_mouse_press(self, event):
		if event.button() == Qt.MiddleButton:
			self.show_inspector()
			return

		pos = (event.position().x(), event.position().y()) if hasattr(event, 'position') else (event.x(), event.y())

		# Right-drag vertical scroll (consistent with other widgets)
		if event.button() == Qt.RightButton:
			self._right_drag_active = True
			self._right_drag_start = pos[1]
			event.accept()
			return

		if self.mmode == "App" and event.button() == Qt.LeftButton:
			img = self._hit_test_image(pos[0], pos[1])
			if img is not None:
				self.mx_image_selected.emit(event, (img,))

		elif self.mmode == "Sets" and event.button() == Qt.LeftButton:
			# Toggle set membership on press so user sees immediate feedback
			img = self._hit_test_image(pos[0], pos[1])
			if img is not None:
				self._ensure_active_set()
				self._toggle_set_membership(img)

		elif self.mmode == "Del" and event.button() == Qt.LeftButton:
			# Toggle deletion - adds/removes from "Deleted" set (always visible, always red)
			img = self._hit_test_image(pos[0], pos[1])
			if img is not None:
				self._toggle_delete(img)

		event.accept()

	def _on_mouse_move(self, event):
		# Right-drag vertical scroll
		if self._right_drag_active:
			pos = (event.position().x(), event.position().y()) if hasattr(event, 'position') else (event.x(), event.y())
			dy = pos[1] - self._right_drag_start
			self.scroll_offset += dy
			min_sc, max_sc = self._compute_scroll_range()
			self.scroll_offset = max(min_sc, min(max_sc, self.scroll_offset))
			self._dirty = True
			self._request_render()
			self._right_drag_start = pos[1]
			event.accept()
			return

		event.accept()

	def _on_mouse_release(self, event):
		if event.button() == Qt.RightButton:
			self._right_drag_active = False
			self._right_drag_start = None

		event.accept()

	def _on_wheel(self, event):
		"""Mouse wheel adjusts the tile scale rather than scrolling."""
		delta = event.angleDelta().y()
		if delta > 0:
			self.scale *= 1.1
		else:
			self.scale /= 1.1
		self.scale = max(0.05, min(self.scale, 50.0))
		self.scroll_offset = 0
		self._dirty = True
		self._request_render()

		if self.inspector:
			self.inspector._sync_from_widget()
		event.accept()

	def _hit_test_image(self, sx, sy):
		"""Find which image index is at the given screen position."""
		if self._data is None or self.nimg == 0:
			return None

		# Compensate for Y-flip caused by offscreen blitting
		y = self.height() - sy
		x = sx

		rowstart, visiblerows, visiblecols, _ = self._compute_grid()
		rw = self._img_xsize * self.scale
		rh = self._img_ysize * self.scale

		for row in range(rowstart, rowstart + visiblerows):
			for col in range(visiblecols):
				idx = row * visiblecols + col
				if idx >= self.nimg:
					break

				tx = col * (rw + self.min_sep) + self.min_sep
				ty = row * (rh + self.min_sep) + self.min_sep - self.scroll_offset

				if tx <= x <= tx + rw and ty <= y <= ty + rh:
					return idx

		return None

	def _ensure_active_set(self):
		"""Ensure there is an active set for membership toggling.

		Creates a 'Default' set if none exist, or selects the first set
		if no set is currently active. Also refreshes inspector list.
		"""
		created = False
		if not self.sets:
			# No sets exist → create Default
			self.enable_set("Default", [], display=True)
			self.current_set = "Default"
			created = True
		elif self.current_set is None:
			# Sets exist but none selected → pick first
			self.current_set = sorted(self.sets.keys())[0]

		if created and self.inspector:
			self.inspector._refresh_set_list()

	def _toggle_set_membership(self, img_idx):
		"""Toggle the clicked image's membership in the current set."""
		if self.current_set is None:
			return
		s = self.get_set(self.current_set)
		if img_idx in s:
			s.discard(img_idx)
		else:
			s.add(img_idx)
		self._clear_scene_groups()
		self._dirty = True
		self._request_render()

	def _toggle_delete(self, img_idx):
		"""Toggle an image's membership in the special "Deleted" set.

		The Deleted set is always visible and always colored bright red (index 0).
		It is automatically created on first use and kept at the top of the list.
		"""
		deleted_set = self.get_set("Deleted")
		if img_idx in deleted_set:
			deleted_set.discard(img_idx)
		else:
			deleted_set.add(img_idx)

		# Ensure Deleted set is always visible
		self.sets_visible["Deleted"] = self.sets["Deleted"]

		self._clear_scene_groups()
		self._dirty = True
		self._request_render()

		# Refresh inspector set list to show Deleted set
		if self.inspector:
			self.inspector._refresh_set_list()

	def remove_particle_image(self, idx):
		"""Remove (exclude) a particle image from display."""
		self._deleted_idxs.add(idx)
		self._dirty = True
		self._request_render()

	def closeEvent(self, event):
		if self.inspector and self.inspector.isVisible():
			self.inspector.close()
		event.accept()

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self._dirty = True
		self._request_render()


# ─── Inspector ──────────────────────────────────────────────────

class EMImageInspectorMX(QtWidgets.QWidget):
	"""Inspector panel for EMImageMXWidget."""

	def __init__(self, target, parent=None):
		super().__init__(parent)

		self._target = weakref.ref(target)
		self.setWindowTitle("EMImageMX Inspector")

		self._build_ui()
		self._wire_signals()
		self._sync_from_widget()

	def _tgt(self):
		return self._target()

	def _build_ui(self):
		main_layout = QtWidgets.QVBoxLayout(self)
		main_layout.setContentsMargins(6, 6, 6, 6)
		main_layout.setSpacing(6)

		# Mouse mode buttons
		mode_layout = QtWidgets.QHBoxLayout()
		self._btn_app = QtWidgets.QPushButton("App")
		self._btn_app.setCheckable(True)
		self._btn_del = QtWidgets.QPushButton("Del")
		self._btn_del.setCheckable(True)
		self._btn_sets = QtWidgets.QPushButton("Sets")
		self._btn_sets.setCheckable(True)

		self._mode_group = QtWidgets.QButtonGroup(self)
		self._mode_group.addButton(self._btn_app)
		self._mode_group.addButton(self._btn_del)
		self._mode_group.addButton(self._btn_sets)
		self._mode_group.setExclusive(True)
		self._btn_app.setChecked(True)

		mode_layout.addWidget(self._btn_app)
		mode_layout.addWidget(self._btn_del)
		mode_layout.addWidget(self._btn_sets)
		main_layout.addLayout(mode_layout)

		# Sets panel
		sets_group = QtWidgets.QGroupBox("Sets")
		sets_layout = QtWidgets.QHBoxLayout(sets_group)

		self._set_list = QtWidgets.QListWidget()
		sets_layout.addWidget(self._set_list)

		set_btn_layout = QtWidgets.QVBoxLayout()
		self._btn_new_set = QtWidgets.QPushButton("New")
		self._btn_new_set.setToolTip("Create a new named set for tagging images")
		self._btn_del_set = QtWidgets.QPushButton("Delete")
		self._btn_del_set.setToolTip("Remove selected sets")
		self._btn_save_set = QtWidgets.QPushButton("Save Set")
		self._btn_save_set.setToolTip("Save set member images to a file")
		self._btn_save_txt = QtWidgets.QPushButton("Save Text")
		self._btn_save_txt.setToolTip("Save set member indices as text")

		set_btn_layout.addWidget(self._btn_new_set)
		set_btn_layout.addWidget(self._btn_del_set)
		set_btn_layout.addWidget(self._btn_save_set)
		set_btn_layout.addWidget(self._btn_save_txt)
		set_btn_layout.addStretch()
		sets_layout.addLayout(set_btn_layout)

		main_layout.addWidget(sets_group)

		self._set_list.itemChanged.connect(self._on_set_item_changed)
		self._set_list.currentRowChanged.connect(self._on_set_row_changed)

		# AutoC / FullC / Invert row
		ctrl_layout = QtWidgets.QHBoxLayout()
		self._btn_autoc = QtWidgets.QPushButton("AutoC")
		self._btn_autoc.setToolTip("Auto-contrast: mean +/- 3 sigma")
		self._btn_fullc = QtWidgets.QPushButton("FullC")
		self._btn_fullc.setToolTip("Full contrast: min to max of all data")
		self._btn_invert = QtWidgets.QPushButton("Invert")
		self._btn_invert.setCheckable(True)
		self._btn_invert.setToolTip("Invert grayscale display")

		ctrl_layout.addWidget(self._btn_autoc)
		ctrl_layout.addWidget(self._btn_fullc)
		ctrl_layout.addWidget(self._btn_invert)
		main_layout.addLayout(ctrl_layout)

		# Display controls
		self._scale = ValSlider(None, (0.05, 10.0), "Scale:")
		self._scale.setValue(1.0)
		self._scale.setToolTip("Magnification of each image tile")
		main_layout.addWidget(self._scale)

		self._min = ValSlider(None, (0, 1), "Min:")
		self._min.setToolTip("Minimum density value for contrast mapping")
		main_layout.addWidget(self._min)

		self._max = ValSlider(None, (0, 1), "Max:")
		self._max.setToolTip("Maximum density value for contrast mapping")
		main_layout.addWidget(self._max)

		self._brt = ValSlider(None, (-1.0, 1.0), "Brt:")
		self._brt.setValue(0.0)
		self._brt.setToolTip("Brightness offset")
		main_layout.addWidget(self._brt)

		self._cont = ValSlider(None, (0, 1.0), "Cont:")
		self._cont.setValue(0.5)
		self._cont.setToolTip("Contrast multiplier")
		main_layout.addWidget(self._cont)

		self._gamma = ValSlider(None, (0.1, 5.0), "Gamma:")
		self._gamma.setValue(1.0)
		self._gamma.setToolTip("Gamma correction")
		main_layout.addWidget(self._gamma)

		# Values dropdown - toggle header keys to display on each image
		self._btn_vals = QtWidgets.QPushButton("Values")
		self._btn_vals.setToolTip("Select header fields to display on images")
		self._vals_menu = QtWidgets.QMenu(self._btn_vals)
		self._btn_vals.setMenu(self._vals_menu)
		self._vals_actions = {}  # key -> QAction
		main_layout.addWidget(self._btn_vals)

		# Action buttons
		btn_layout = QtWidgets.QHBoxLayout()
		self._btn_snapshot = QtWidgets.QPushButton("Snapshot")
		self._btn_snapshot.setToolTip("Save screenshot of current view")
		self._btn_save = QtWidgets.QPushButton("Save Data")
		self._btn_save.setToolTip("Export displayed images to file")
		self._btn_open2d = QtWidgets.QPushButton("Open 2D")
		self._btn_open2d.setToolTip("Open selected image in single-image viewer")

		btn_layout.addWidget(self._btn_snapshot)
		btn_layout.addWidget(self._btn_save)
		btn_layout.addWidget(self._btn_open2d)
		main_layout.addLayout(btn_layout)

	def _wire_signals(self):
		tgt = self._tgt()
		if not tgt:
			return

		self._mode_group.buttonClicked.connect(self._on_mode_change)
		self._scale.valueChanged.connect(self._on_scale_changed)
		self._min.valueChanged.connect(self._on_min_changed)
		self._max.valueChanged.connect(self._on_max_changed)
		self._brt.valueChanged.connect(self._on_brt_changed)
		self._cont.valueChanged.connect(self._on_cont_changed)
		self._gamma.valueChanged.connect(self._on_gamma_changed)

		self._btn_snapshot.clicked.connect(self._on_snapshot)
		self._btn_save.clicked.connect(self._on_save_data)
		self._btn_open2d.clicked.connect(self._on_open_2d)

		self._btn_new_set.clicked.connect(self._on_new_set)
		self._btn_del_set.clicked.connect(self._on_delete_set)
		self._btn_save_set.clicked.connect(self._on_save_set)
		self._btn_save_txt.clicked.connect(self._on_save_set_text)

		self._btn_autoc.clicked.connect(self._on_auto_contrast)
		self._btn_fullc.clicked.connect(self._on_full_contrast)
		self._btn_invert.toggled.connect(self._on_invert_toggled)

	def _refresh_vals_menu(self):
		"""Populate the Values dropdown from available header keys."""
		tgt = self._tgt()
		if not tgt or not tgt._data_headers:
			return

		# Collect unique header keys from first valid header
		keys = []
		for h in tgt._data_headers:
			if isinstance(h, dict) and h:
				keys = sorted(h.keys())
				break

		self._vals_menu.clear()
		self._vals_actions.clear()

		# Always include "Img #"
		img_action = self._vals_menu.addAction("Img #")
		img_action.setCheckable(True)
		img_action.setChecked("Img #" in tgt.valstodisp)
		self._vals_actions["Img #"] = img_action
		img_action.toggled.connect(lambda checked, k="Img #": self._on_val_toggle(k, checked))

		for k in keys:
			action = self._vals_menu.addAction(str(k))
			action.setCheckable(True)
			action.setChecked(k in tgt.valstodisp)
			self._vals_actions[k] = action
			action.toggled.connect(lambda checked, k=k: self._on_val_toggle(str(k), checked))

	def _on_val_toggle(self, key, checked):
		"""Toggle a header key in the display list."""
		tgt = self._tgt()
		if not tgt:
			return
		if checked and key not in tgt.valstodisp:
			tgt.valstodisp.append(key)
		elif not checked and key in tgt.valstodisp:
			tgt.valstodisp.remove(key)
		tgt._dirty = True
		tgt._request_render()


	def _sync_from_widget(self):
		"""Pull current values from the widget into inspector controls."""
		tgt = self._tgt()
		if not tgt:
			return

		self._scale.setValue(tgt.scale)
		
		# Min/Max sliders show current contrast values (curmin/curmax)
		self._min.setValue(tgt.curmin)
		self._max.setValue(tgt.curmax)
		self._gamma.setValue(tgt.gamma)

		# Slider ranges bounded by full data range (minden/maxden)
		self._min.setRange(tgt.minden, tgt.maxden)
		self._max.setRange(tgt.minden, tgt.maxden)

		self._btn_invert.setChecked(tgt.invert)

		# Compute Brt/Cont back from curmin/curmax within global range (minden/maxden)
		# Same formulas as EMImage2D inspector _update_brightness_contrast
		range_diff = tgt.maxden - tgt.minden
		if abs(range_diff) > 1e-12:
			min_val = tgt.curmin
			max_val = tgt.curmax

			b = 0.5 * (min_val + max_val - (tgt.minden + tgt.maxden)) / range_diff
			c = (min_val - max_val) / (2.0 * (tgt.minden - tgt.maxden))
			brts = -b
			conts = 1.0 - c

			self._brt.setValue(max(-1.0, min(1.0, brts)), quiet=1)
			self._cont.setValue(max(0.0, min(1.0, conts)), quiet=1)

		mode = tgt.mmode
		btn_map = {
			"App": self._btn_app, "Del": self._btn_del,
			"Sets": self._btn_sets,
		}
		if mode in btn_map:
			btn_map[mode].setChecked(True)

		self._refresh_set_list()
		self._refresh_vals_menu()

	def _refresh_set_list(self):
		tgt = self._tgt()
		if not tgt:
			return

		self._set_list.clear()
		keys = sorted(tgt.sets.keys())
		vis_keys = set(tgt.sets_visible.keys())

		# Build color map: "Deleted" gets 0 (red), others shifted +1
		base_item_flags = (Qt.ItemIsSelectable | Qt.ItemIsEnabled |
		              Qt.ItemIsUserCheckable)
		color_map = {}
		nd_count = 0
		for k in keys:
			if k == "Deleted":
				color_map[k] = 0
			else:
				color_map[k] = nd_count + 1
				nd_count += 1

		for i, k in enumerate(keys):
			item = QtWidgets.QListWidgetItem(k)
			# Deleted set is always visible and can't be unchecked
			if k == "Deleted":
				item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
			else:
				item.setFlags(base_item_flags)

			c = SET_COLORS[color_map[k] % len(SET_COLORS)]
			q_color = QtGui.QColor()
			q_color.setRgbF(c[0], c[1], c[2], c[3])
			item.setForeground(q_color)

			if k in vis_keys:
				item.setCheckState(Qt.Checked)
			else:
				item.setCheckState(Qt.Unchecked)

			self._set_list.addItem(item)

	# ─── Signal handlers ────────────────────────────────────────

	def _on_mode_change(self, button):
		tgt = self._tgt()
		if tgt:
			tgt.set_mouse_mode(button.text())

	def _on_scale_changed(self, val):
		tgt = self._tgt()
		if tgt:
			tgt.set_scale(val)

	def _on_min_changed(self, val):
		"""Min slider changed - update curmin and Brt/Cont."""
		tgt = self._tgt()
		if not tgt:
			return
		tgt.curmin = float(val)
		tgt._clear_scene_groups()
		tgt._dirty = True
		tgt._request_render()
		# Sync Brt/Cont back from curmin/curmax
		range_diff = tgt.maxden - tgt.minden
		if abs(range_diff) > 1e-12:
			min_val = tgt.curmin
			max_val = tgt.curmax
			b = 0.5 * (min_val + max_val - (tgt.minden + tgt.maxden)) / range_diff
			c = (min_val - max_val) / (2.0 * (tgt.minden - tgt.maxden))
			self._brt.setValue(max(-1.0, min(1.0, -b)), quiet=1)
			self._cont.setValue(max(0.0, min(1.0, 1.0 - c)), quiet=1)

	def _on_max_changed(self, val):
		"""Max slider changed - update curmax and Brt/Cont."""
		tgt = self._tgt()
		if not tgt:
			return
		tgt.curmax = float(val)
		tgt._clear_scene_groups()
		tgt._dirty = True
		tgt._request_render()
		# Sync Brt/Cont back from curmin/curmax
		range_diff = tgt.maxden - tgt.minden
		if abs(range_diff) > 1e-12:
			min_val = tgt.curmin
			max_val = tgt.curmax
			b = 0.5 * (min_val + max_val - (tgt.minden + tgt.maxden)) / range_diff
			c = (min_val - max_val) / (2.0 * (tgt.minden - tgt.maxden))
			self._brt.setValue(max(-1.0, min(1.0, -b)), quiet=1)
			self._cont.setValue(max(0.0, min(1.0, 1.0 - c)), quiet=1)

	def _on_brt_changed(self, val):
		"""Brightness changed - update curmin/curmax from Brt/Cont."""
		tgt = self._tgt()
		if not tgt:
			return
		self._update_curmin_max_from_slider(tgt)

	def _on_cont_changed(self, val):
		"""Contrast changed - update curmin/curmax from Brt/Cont."""
		tgt = self._tgt()
		if not tgt:
			return
		self._update_curmin_max_from_slider(tgt)

	def _update_curmin_max_from_slider(self, tgt):
		"""Convert Brt/Cont slider values to curmin/curmax. Same as EMImage2D _update_min_max."""
		range_diff = tgt.maxden - tgt.minden
		if abs(range_diff) < 1e-12:
			return

		center = (tgt.minden + tgt.maxden) / 2.0
		brt_val = self._brt.getValue()
		cont_val = self._cont.getValue()

		x0 = center - range_diff * (1.0 - cont_val) - brt_val * range_diff
		x1 = center + range_diff * (1.0 - cont_val) - brt_val * range_diff

		tgt.curmin = max(tgt.minden, min(tgt.maxden, x0))
		tgt.curmax = max(tgt.minden, min(tgt.maxden, x1))

		# Update Min/Max slider values without triggering their handlers
		self._min.setValue(tgt.curmin, quiet=1)
		self._max.setValue(tgt.curmax, quiet=1)

		tgt._clear_scene_groups()
		tgt._dirty = True
		tgt._request_render()

	def _on_gamma_changed(self, val):
		tgt = self._tgt()
		if tgt:
			tgt.set_gamma(val)

	def _on_auto_contrast(self):
		"""Auto-contrast: mean +/- 3 sigma sampled across images."""
		tgt = self._tgt()
		if tgt:
			tgt.auto_contrast()
			self._sync_from_widget()

	def _on_full_contrast(self):
		"""Full contrast: global min to max of all data."""
		tgt = self._tgt()
		if tgt:
			tgt.full_contrast()
			self._sync_from_widget()

	def _on_invert_toggled(self, checked):
		"""Toggle grayscale inversion."""
		tgt = self._tgt()
		if tgt:
			tgt.set_invert(checked)

	def _on_snapshot(self):
		fsp, _ = QtWidgets.QFileDialog.getSaveFileName(
			self, "Save Snapshot", "", "Images (*.png *.jpg)"
		)
		if fsp:
			tgt = self._tgt()
			if tgt and hasattr(tgt, '_canvas'):
				pixmap = tgt._canvas.grab()
				pixmap.save(fsp)

	def _on_save_data(self):
		tgt = self._tgt()
		if not tgt or not tgt._data:
			return
		fsp, _ = QtWidgets.QFileDialog.getSaveFileName(
			self, "Save Data", "", "Numpy files (*.npy)"
		)
		if not fsp:
			return

		# Get deleted indices (exclude from save)
		deleted_idxs = set()
		if "Deleted" in tgt.sets:
			deleted_idxs = tgt.sets["Deleted"]

		# Collect non-deleted images
		saved = []
		for i, d in enumerate(tgt._data):
			if d is not None and i not in deleted_idxs:
				saved.append(np.asarray(d))

		if saved:
			stacked = np.stack(saved)
			np.save(fsp, stacked)
			n_deleted = len(deleted_idxs)
			print(f"Saved {len(saved)} images to {fsp}" + 
			      (f" ({n_deleted} excluded)" if n_deleted else ""))

	def _on_open_2d(self):
		tgt = self._tgt()
		if tgt and tgt._data and tgt.nimg > 0:
			try:
				from EMAN3.gui.emimage2d import EMImage2DWidget
				viewer = EMImage2DWidget()
				viewer.set_data(tgt._data)
				viewer.show()
			except Exception as e:
				print(f"Could not open 2D viewer: {e}")

	def _on_new_set(self):
		name, ok = QtWidgets.QInputDialog.getText(
			self, "New Set", "Enter a name for the new set:"
		)
		if not (ok and name):
			return
		if name == "Deleted":
			print("Set name 'Deleted' is reserved - use Del mouse mode instead.")
			return
		tgt = self._tgt()
		if tgt and name not in tgt.sets:
			tgt.enable_set(name, [], display=True)
			self._refresh_set_list()

	def _on_delete_set(self):
		selected = self._set_list.selectedItems()
		names = [str(item.text()) for item in selected]
		# Prevent deleting the "Deleted" set - it can be emptied via Del mode, not removed
		names = [n for n in names if n != "Deleted"]
		if not names:
			return
		tgt = self._tgt()
		if tgt:
			tgt.delete_set(names)
			self._refresh_set_list()

	def _on_save_set(self):
		selected = self._set_list.selectedItems()
		tgt = self._tgt()
		if not tgt or not selected:
			return
		for item in selected:
			name = str(item.text())
			s = tgt.get_set(name)
			fsp, _ = QtWidgets.QFileDialog.getSaveFileName(
				self, "Save Set", "", "Numpy files (*.npy)"
			)
			if fsp:
				idxs = sorted(s)
				data = [tgt._data[i] for i in idxs if 0 <= i < len(tgt._data)]
				if data:
					stacked = np.stack([np.asarray(d) for d in data])
					np.save(fsp, stacked)

	def _on_save_set_text(self):
		selected = self._set_list.selectedItems()
		tgt = self._tgt()
		if not tgt or not selected:
			return
		for item in selected:
			name = str(item.text())
			s = tgt.get_set(name)
			fsp, _ = QtWidgets.QFileDialog.getSaveFileName(
				self, "Save Set Text", "", "Text files (*.txt)"
			)
			if fsp:
				with open(fsp, 'w') as f:
					for i in sorted(s):
						f.write(f"{i}\n")

	def _on_set_item_changed(self, item):
		tgt = self._tgt()
		if not tgt:
			return
		name = str(item.text())
		# Prevent unchecking the "Deleted" set - it must always be visible
		if name == "Deleted" and item.checkState() == Qt.Unchecked:
			item.setCheckState(Qt.Checked)
			return
		if item.checkState() == Qt.Checked:
			tgt.show_set(name)
		else:
			tgt.hide_set(name)

	def _on_set_row_changed(self, row):
		tgt = self._tgt()
		if not tgt or row < 0:
			return
		item = self._set_list.item(row)
		if item:
			name = str(item.text())
			tgt.current_set = name

	def closeEvent(self, event):
		event.accept()


def main():
	"""Test program for EMImageMXWidget.

	Usage:
	    python emimagemx.py [image_stack_file]

	If no image file is provided, generates 50 random test images (64x64).
	Otherwise loads the given image stack as a list of 2D slices.
	"""
	import sys

	app = QtWidgets.QApplication(sys.argv)

	widget = EMImageMXWidget()

	if len(sys.argv) >= 2:
		image_path = sys.argv[1]
		try:
			from EMAN3.io.imageio import ImageIO
			io = ImageIO(image_path, "r")
			nimg = io.nimg
			print(f"Loaded: {image_path} ({nimg} image(s))")

			if nimg > 1:
				data, headers = io.read_images()
				stack_data = [data[i] for i in range(data.shape[0])]
			else:
				data, header = io.read_image(0)
				print(f"Read image: {data.shape}, header keys: {list(header.keys())}")
				# If 3D, take first z-slice or convert to list of slices
				if data.ndim == 3:
					stack_data = [data[i] for i in range(data.shape[0])]
				else:
					stack_data = [data]

			widget.set_data(stack_data, filename=image_path)
		except Exception as e:
			import traceback
			traceback.print_exc()
			print(f"Error loading file: {e}")
			return
	else:
		# Generate test data: random 64x64 images
		np.random.seed(42)
		data = []
		for i in range(50):
			img = np.random.rand(64, 64).astype(np.float32)
			data.append(img)

		widget.set_data(data, filename="test_multiimage")

	widget.resize(800, 700)
	widget.show()

	print("Middle-click to show inspector.")
	print("Scroll wheel to adjust tile scale.")
	print("Right-drag to scroll the grid vertically.")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
