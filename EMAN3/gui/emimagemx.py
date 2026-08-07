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
SET_COLORS = [
	(0.0, 0.0, 1.0, 1.0),       # blue
	(0.0, 0.85, 0.0, 1.0),      # green
	(1.0, 0.0, 0.0, 1.0),       # red
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
		self.invert = False
		self.gamma = 1.0
		self.brightness = 0.0
		self.contrast = 1.0

		# Grid layout
		self.min_sep = 4             # min separation between tiles (px)
		self.scroll_offset = 0       # vertical scroll offset in px

		# Mouse mode
		self.mouse_modes = ["App", "Del", "Drag", "Sets"]
		self.mmode = "App"
		self._drag_start_pos = None
		self._drag_scroll_at_start = None
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
		ty = row * (rh + self.min_sep) + self.min_sep + self.scroll_offset

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
				self.minden, self.maxden,
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
				ty = row * (rh + self.min_sep) + self.min_sep + self.scroll_offset

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
				ty = row * (rh + self.min_sep) + self.min_sep + self.scroll_offset

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
		return " ".join(parts)

	# ─── Data handling ──────────────────────────────────────────

	def set_data(self, obj, filename="", metadata=None):
		"""Set the data to display.

		Accepts a list of numpy 2D arrays (float), EMData objects, or
		an iterable of images.
		"""
		self.file_name = filename
		if filename:
			self.setWindowTitle(os.path.basename(filename))

		images = []
		headers = []

		if isinstance(obj, (list, tuple)):
			for item in obj:
				if hasattr(item, 'get_data'):
					# EMData-like object
					arr = self._emdata_to_numpy(item)
					images.append(arr)
					headers.append(self._emdata_header(item))
				elif isinstance(item, np.ndarray):
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
			try:
				if hasattr(obj, 'nproc'):
					for i in range(len(obj)):
						frame = obj[i]
						images.append(self._emdata_to_numpy(frame))
						headers.append(self._emdata_header(frame))
			except Exception:
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

		self._auto_contrast()

		self._dirty = True
		self._request_render()

	def _emdata_to_numpy(self, emd):
		"""Convert an EMData-like object to a 2D numpy float32 array."""
		try:
			arr = emd.get_data(process=False)
		except AttributeError:
			try:
				arr = np.array(emd)
			except Exception:
				return None

		arr = arr.astype(np.float32)
		if arr.ndim == 3:
			if arr.shape[0] == 1:
				arr = arr[0]
			else:
				min_dim = arr.shape.index(min(arr.shape))
				arr = np.take(arr, 0, axis=min_dim)

		return arr

	def _emdata_header(self, emd):
		"""Extract header metadata from an EMData-like object."""
		header = {}
		try:
			keys = emd.get_all_headers()
			for key in keys:
				try:
					val = emd.get_attr(key)
					header[key] = val
				except Exception:
					pass
		except Exception:
			pass
		return header

	def _auto_contrast(self):
		"""Compute automatic contrast range by sampling images."""
		if not self._data or self.nimg == 0:
			self.minden = 0.0
			self.maxden = 1.0
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
			self.minden = 0.0
			self.maxden = 1.0
			return

		mean_sum /= count
		self.minden = max(global_min, mean_sum - 3.0 * max_sigma)
		self.maxden = min(global_max, mean_sum + 4.0 * max_sigma)

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
		self.minden = val
		self._dirty = True
		self._request_render()

	def get_density_min(self):
		return self.minden

	def set_density_max(self, val):
		self.maxden = val
		self._dirty = True
		self._request_render()

	def get_density_max(self):
		return self.maxden

	def set_den_range(self, mn, mx):
		self.minden = mn
		self.maxden = mx
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

	def auto_contrast(self):
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
		count = 0

		for i in range(0, len(valid), stp):
			img = valid[i]
			if img is None:
				continue
			mean_sum += float(np.mean(img))
			count += 1
			max_sigma = max(max_sigma, float(np.std(img)))

		if count == 0:
			return

		mean_sum /= count
		# Use global min/max from first sample pass
		global_min = min(float(np.min(valid[i])) for i in range(0, len(valid), stp) if valid[i] is not None)
		global_max = max(float(np.max(valid[i])) for i in range(0, len(valid), stp) if valid[i] is not None)

		self.minden = max(global_min, mean_sum - 3.0 * max_sigma)
		self.maxden = min(global_max, mean_sum + 4.0 * max_sigma)

		self._dirty = True
		self._request_render()

	def full_contrast(self):
		"""Full contrast: global min to max of all data."""
		if not self._data or self.nimg == 0:
			return

		valid = [img for img in self._data if img is not None]
		if not valid:
			return

		n_sample = min(len(valid), 64)
		stp = max(len(valid) // n_sample, 1)

		global_min = float('inf')
		global_max = float('-inf')

		for i in range(0, len(valid), stp):
			img = valid[i]
			if img is None:
				continue
			global_min = min(global_min, float(np.min(img)))
			global_max = max(global_max, float(np.max(img)))

		self.minden = global_min
		self.maxden = global_max

		self._dirty = True
		self._request_render()

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

		elif self.mmode == "Drag":
			self._drag_start_pos = pos
			self._drag_scroll_at_start = self.scroll_offset

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

		if self.mmode == "Drag" and self._drag_start_pos:
			pos = (event.position().x(), event.position().y()) if hasattr(event, 'position') else (event.x(), event.y())
			dy = pos[1] - self._drag_start_pos[1]
			self.scroll_offset = self._drag_scroll_at_start + dy
			min_sc, max_sc = self._compute_scroll_range()
			self.scroll_offset = max(min_sc, min(max_sc, self.scroll_offset))
			self._dirty = True
			self._request_render()

		event.accept()

	def _on_mouse_release(self, event):
		if event.button() == Qt.RightButton:
			self._right_drag_active = False
			self._right_drag_start = None

		if self.mmode == "Drag":
			self._drag_start_pos = None

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
				ty = row * (rh + self.min_sep) + self.min_sep + self.scroll_offset

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
		self._btn_drag = QtWidgets.QPushButton("Drag")
		self._btn_drag.setCheckable(True)
		self._btn_sets = QtWidgets.QPushButton("Sets")
		self._btn_sets.setCheckable(True)

		self._mode_group = QtWidgets.QButtonGroup(self)
		self._mode_group.addButton(self._btn_app)
		self._mode_group.addButton(self._btn_del)
		self._mode_group.addButton(self._btn_drag)
		self._mode_group.addButton(self._btn_sets)
		self._mode_group.setExclusive(True)
		self._btn_app.setChecked(True)

		mode_layout.addWidget(self._btn_app)
		mode_layout.addWidget(self._btn_del)
		mode_layout.addWidget(self._btn_drag)
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

	def _sync_from_widget(self):
		"""Pull current values from the widget into inspector controls."""
		tgt = self._tgt()
		if not tgt:
			return

		self._scale.setValue(tgt.scale)
		self._min.setValue(tgt.minden)
		self._max.setValue(tgt.maxden)
		self._gamma.setValue(tgt.gamma)

		self._min.setRange(tgt.minden - 0.5, tgt.maxden + 0.5)
		self._max.setRange(tgt.minden - 0.5, tgt.maxden + 0.5)

		self._btn_invert.setChecked(tgt.invert)

		mode = tgt.mmode
		btn_map = {
			"App": self._btn_app, "Del": self._btn_del,
			"Drag": self._btn_drag, "Sets": self._btn_sets,
		}
		if mode in btn_map:
			btn_map[mode].setChecked(True)

		self._refresh_set_list()

	def _refresh_set_list(self):
		tgt = self._tgt()
		if not tgt:
			return

		self._set_list.clear()
		keys = sorted(tgt.sets.keys())
		vis_keys = set(tgt.sets_visible.keys())

		item_flags = (Qt.ItemIsSelectable | Qt.ItemIsEnabled |
		              Qt.ItemIsUserCheckable)

		for i, k in enumerate(keys):
			item = QtWidgets.QListWidgetItem(k)
			item.setFlags(item_flags)

			color_i = i % len(SET_COLORS)
			c = SET_COLORS[color_i]
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
		tgt = self._tgt()
		if tgt:
			tgt.set_density_min(val)

	def _on_max_changed(self, val):
		tgt = self._tgt()
		if tgt:
			tgt.set_density_max(val)

	def _on_brt_changed(self, val):
		pass  # Brightness handled in render pipeline

	def _on_cont_changed(self, val):
		pass  # Contrast handled in render pipeline

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
		if tgt and tgt._data:
			fsp, _ = QtWidgets.QFileDialog.getSaveFileName(
				self, "Save Data", "", "Numpy files (*.npy)"
			)
			if fsp:
				stacked = np.stack([np.asarray(d) for d in tgt._data if d is not None])
				np.save(fsp, stacked)

	def _on_open_2d(self):
		tgt = self._tgt()
		if tgt and tgt._data and tgt.nimg > 0:
			try:
				from EMAN3.gui.emimage2d import EMImage2DWidget
				viewer = EMImage2DWidget()
				viewer.set_data(tgt._data[0])
				viewer.show()
			except Exception as e:
				print(f"Could not open 2D viewer: {e}")

	def _on_new_set(self):
		name, ok = QtWidgets.QInputDialog.getText(
			self, "New Set", "Enter a name for the new set:"
		)
		if ok and name:
			tgt = self._tgt()
			if tgt and name not in tgt.sets:
				tgt.enable_set(name, [], display=True)
				self._refresh_set_list()

	def _on_delete_set(self):
		selected = self._set_list.selectedItems()
		names = [str(item.text()) for item in selected]
		tgt = self._tgt()
		if tgt and names:
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
	"""Test program for EMImageMXWidget."""
	import sys

	app = QtWidgets.QApplication(sys.argv)

	widget = EMImageMXWidget()

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
	print("Scroll wheel to scroll through images.")
	print("Drag mode: left-drag to scroll the grid.")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
