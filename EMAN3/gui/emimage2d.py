# EMAN3 2D Image Viewer - PyGfx rendering with PySide6
# Ported from EMAN2 qtgui/emimage2d.py

import weakref
from EMAN3.gui.valslider import ValSlider, ValBox, StringBox

import pygfx as gfx
from rendercanvas.qt import QRenderWidget

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from EMAN3.gui.emshape import (
	ShapeLine, ShapeCircle, ShapeLabel,
	ShapeScrRect, ShapeScrLabel, ShapeRCircle,
	ShapeScrCircle, ShapeRectPoint,
)
import numpy as np

# ─── NumPy rendering helpers (replacing GLUtil C++) ──────────────────────

def _render_image_8bit(data, scale, origin_x, origin_y,
		       display_w, display_h,
		       min_val, max_val, gamma=1.0,
		       invert=False, histogram_mode=0):
	"""Render a float32 2D image to float32 RGB [0..1] for PyGfx texture.

	origin = (origin_x, origin_y) is the screen position (from window top-left)
	where data pixel (0, 0) should appear. Negative means off-screen.

	Returns (rgb_image_2d, hist_counts) where rgb_image is (H,W,3) float32.
	"""
	if data is None:
		return np.zeros((display_h, display_w, 3), dtype=np.float32), None

	data = np.asarray(data, dtype=np.float32)
	ny, nx = data.shape

	inv_scale = 1.0 / scale

	# Which data pixels are visible on screen?
	src_x0 = max(0, int(np.floor(-origin_x * inv_scale)))
	src_y0 = max(0, int(np.floor(-origin_y * inv_scale)))
	src_x1 = min(nx, int(np.ceil((display_w - origin_x) * inv_scale)))
	src_y1 = min(ny, int(np.ceil((display_h - origin_y) * inv_scale)))

	if src_x1 <= src_x0 or src_y1 <= src_y0:
		return np.zeros((display_h, display_w, 3), dtype=np.float32), None

	# Extract visible region (no Y-flip: numpy row 0 maps to top of screen)
	region = data[src_y0:src_y1, src_x0:src_x1]

	actual_h = src_y1 - src_y0
	actual_w = src_x1 - src_x0

	rendered_h = max(1, int(round(actual_h * scale)))
	rendered_w = max(1, int(round(actual_w * scale)))

	if scale > 1.0:
		scaled_region = _resize_nearest(region, rendered_h, rendered_w)
	else:
		scaled_region = _resize_bilinear(region, rendered_h, rendered_w)

	# Place at correct screen position (NOT centered)
	padded = np.full((display_h, display_w), min_val, dtype=np.float32)
	place_x0 = max(0, int(round(origin_x + src_x0 * scale)))
	place_y0 = max(0, int(round(origin_y + src_y0 * scale)))
	sx = min(place_x0 + rendered_w, display_w)
	sy = min(place_y0 + rendered_h, display_h)
	cropped_h = min(sy - place_y0, scaled_region.shape[0])
	cropped_w = min(sx - place_x0, scaled_region.shape[1])
	padded[place_y0:sy, place_x0:sx] = scaled_region[:cropped_h, :cropped_w]

	# Compute histogram before contrast mapping
	hist = None
	if histogram_mode > 0:
		clipped = np.clip(padded, min_val, max_val)
		binned = ((clipped - min_val) / (max_val - min_val) * 255).astype(np.int32)
		binned = np.clip(binned, 0, 255)
		hist, _ = np.histogram(binned.ravel(), bins=256, range=(0, 256))

	# Apply contrast mapping: normalize to [0..1], clip, gamma correct
	if max_val > min_val:
		normed = (padded - min_val) / (max_val - min_val)
	else:
		normed = np.zeros_like(padded)

	normed = np.clip(normed, 0.0, 1.0)

	if gamma != 1.0 and gamma > 0:
		normed = np.power(normed, 1.0 / gamma)

	if invert:
		normed = 1.0 - normed

	# Grayscale -> RGB (float32 in [0..1] range for PyGfx texture)
	rgb = np.stack([normed, normed, normed], axis=-1)

	return rgb, hist


def _resize_bilinear(src, dst_h, dst_w):
	"""Bilinear resize of a 2D float array."""
	if src.shape[0] == dst_h and src.shape[1] == dst_w:
		return src.copy()

	sy, sx = src.shape
	try:
		from scipy.ndimage import zoom
		factor_y = dst_h / sy
		factor_x = dst_w / sx
		return zoom(src, (factor_y, factor_x), order=1)
	except ImportError:
		y_idx = np.linspace(0, sy - 1, dst_h, dtype=np.float32).round().astype(int)
		x_idx = np.linspace(0, sx - 1, dst_w, dtype=np.float32).round().astype(int)
		y_idx = np.clip(y_idx, 0, sy - 1)
		x_idx = np.clip(x_idx, 0, sx - 1)
		return src[np.ix_(y_idx, x_idx)]


def _resize_nearest(src, dst_h, dst_w):
	"""Nearest-neighbor resize of a 2D float array (sharp pixel rendering)."""
	if src.shape[0] == dst_h and src.shape[1] == dst_w:
		return src.copy()

	sy, sx = src.shape
	# Map each output pixel to the nearest source pixel center
	y_idx = np.floor(np.linspace(0.5, sy - 0.5, dst_h)).astype(int)
	x_idx = np.floor(np.linspace(0.5, sx - 0.5, dst_w)).astype(int)
	y_idx = np.clip(y_idx, 0, sy - 1)
	x_idx = np.clip(x_idx, 0, sx - 1)
	return src[np.ix_(y_idx, x_idx)]


def _render_fft_complex_rgb(data_complex, scale, origin_x, origin_y,
			    display_w, display_h,
			    min_val, max_val, gamma=1.0,
			    invert=False):
	"""Render complex FFT data as colored RGB [0..1] float32 for PyGfx texture.

	Uses the same extraction/placement as _render_image_8bit so zoom and pan work.
	Returns (rgb, hist) tuple for consistency with other renderers.
	"""
	if data_complex is None:
		return np.zeros((display_h, display_w, 3), dtype=np.float32), None

	data = np.asarray(data_complex, dtype=np.complex64)
	real_part = data.real
	imag_part = data.imag
	mag = np.sqrt(real_part**2 + imag_part**2)
	ny, nx = real_part.shape

	inv_scale = 1.0 / scale
	src_x0 = max(0, int(np.floor(-origin_x * inv_scale)))
	src_y0 = max(0, int(np.floor(-origin_y * inv_scale)))
	src_x1 = min(nx, int(np.ceil((display_w - origin_x) * inv_scale)))
	src_y1 = min(ny, int(np.ceil((display_h - origin_y) * inv_scale)))

	if src_x1 <= src_x0 or src_y1 <= src_y0:
		return np.zeros((display_h, display_w, 3), dtype=np.float32), None

	real_region = real_part[src_y0:src_y1, src_x0:src_x1]
	imag_region = imag_part[src_y0:src_y1, src_x0:src_x1]
	mag_region = mag[src_y0:src_y1, src_x0:src_x1]

	actual_h = src_y1 - src_y0
	actual_w = src_x1 - src_x0
	rendered_h = max(1, int(round(actual_h * scale)))
	rendered_w = max(1, int(round(actual_w * scale)))

	if scale > 1.0:
		real_scaled = _resize_nearest(real_region, rendered_h, rendered_w)
		imag_scaled = _resize_nearest(imag_region, rendered_h, rendered_w)
		mag_scaled = _resize_nearest(mag_region, rendered_h, rendered_w)
	else:
		real_scaled = _resize_bilinear(real_region, rendered_h, rendered_w)
		imag_scaled = _resize_bilinear(imag_region, rendered_h, rendered_w)
		mag_scaled = _resize_bilinear(mag_region, rendered_h, rendered_w)

	# Normalize magnitude for brightness/saturation
	if max_val > min_val:
		mag_norm = (mag_scaled - min_val) / (max_val - min_val)
	else:
		mag_norm = np.zeros_like(mag_scaled)
	mag_norm = np.clip(mag_norm, 0.0, 1.0)

	if gamma != 1.0 and gamma > 0:
		mag_norm = np.power(mag_norm, 1.0 / gamma)
	if invert:
		mag_norm = 1.0 - mag_norm

	angle = np.arctan2(imag_scaled, real_scaled)
	phase_norm = (angle + np.pi) / (2 * np.pi)
	hue = phase_norm * 255
	sat = mag_norm * 255
	val = mag_norm * 255

	fft_rgb = _hsv_to_rgb(hue, sat, val).astype(np.uint8).astype(np.float32) / 255.0

	# Place into display canvas with correct positioning (same as _render_image_8bit)
	padded = np.zeros((display_h, display_w, 3), dtype=np.float32)
	place_x0 = max(0, int(round(origin_x + src_x0 * scale)))
	place_y0 = max(0, int(round(origin_y + src_y0 * scale)))
	sx = min(place_x0 + rendered_w, display_w)
	sy = min(place_y0 + rendered_h, display_h)
	cropped_h = min(sy - place_y0, fft_rgb.shape[0])
	cropped_w = min(sx - place_x0, fft_rgb.shape[1])
	padded[place_y0:sy, place_x0:sx] = fft_rgb[:cropped_h, :cropped_w]

	return padded, None


def _hsv_to_rgb(h, s, v):
	"""Convert HSV arrays (0-255) to RGB (0-255)."""
	h = np.asarray(h, dtype=np.float32) / 255.0
	s = np.asarray(s, dtype=np.float32) / 255.0
	v = np.asarray(v, dtype=np.float32) / 255.0

	h_i = (h * 6).astype(np.int32) % 6
	f = h * 6 - h_i
	p = v * (1 - s)
	q = v * (1 - s * f)
	t = v * (1 - s * (1 - f))

	r = np.zeros_like(v)
	g = np.zeros_like(v)
	b = np.zeros_like(v)

	masks = [h_i == 0, h_i == 1, h_i == 2, h_i == 3, h_i == 4, h_i == 5]
	r[masks[0]] = v[masks[0]]; g[masks[0]] = t[masks[0]]
	r[masks[1]] = q[masks[1]]; g[masks[1]] = v[masks[1]]
	r[masks[2]] = p[masks[2]]; g[masks[2]] = v[masks[2]]
	r[masks[3]] = p[masks[3]]; b[masks[3]] = t[masks[3]]
	r[masks[4]] = t[masks[4]]; b[masks[4]] = v[masks[4]]
	g[masks[5]] = q[masks[5]]; b[masks[5]] = v[masks[5]]

	return np.stack([r, g, b], axis=-1) * 255


# ─── PyGfx Canvas with Mouse Event Forwarding ────────────────────────────

class _EMCanvas(QRenderWidget):
	"""QRenderWidget subclass that forwards mouse/keyboard/wheel events to parent."""

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


# ─── Image Viewer Widget ────────────────────────────────────────────────

class EMImage2DWidget(QtWidgets.QWidget):
	"""2D image viewer using PyGfx for rendering."""

	origin_update = QtCore.Signal(tuple)
	signal_set_scale = QtCore.Signal(float)
	mousedown = QtCore.Signal(object, tuple)
	mousedrag = QtCore.Signal(object, tuple)
	mousemove = QtCore.Signal(object, tuple)
	mouseup = QtCore.Signal(object, tuple)
	keypress = QtCore.Signal(object)
	signal_increment_list_data = QtCore.Signal(int)

	def __init__(self, stack=None, metadata=None, parent=None):
		super().__init__(parent)

		self.setFocusPolicy(Qt.StrongFocus)
		self.setMouseTracking(True)
		self.setMinimumSize(128, 128)

		# ── Data state ──
		self._stack = stack
		self._metadata = metadata or {}
		self._data_array = None
		self._list_data = None
		self._list_idx = 0
		self._file_name = ""

		# ── Display params ──
		self.scale = 1.0
		self.origin = (0, 0)
		self.invert = False
		self.gamma = 1.0
		self.contrast = 1.0
		self.brightness = 0.0
		self.minden = 0.0
		self.maxden = 1.0
		self.curmin = 0.0
		self.curmax = 1.0
		self.fgamma = 1.0
		self.fminden = 0.0
		self.fmaxden = 1.0
		self.fcurmin = 0.0
		self.fcurmax = 1.0

		# ── FFT state ──
		self.curfft = 0
		self._display_fft = None
		self._fft_cache = {}
		self.fftorigincenter = False

		# ── Histogram mode ──
		self.histogram_mode = 0

		# ── Mouse mode ──
		self._mouse_modes = {
			0: "emit", 1: "emit", 2: "emit",
			3: "probe", 4: "measure", 5: "draw",
			6: "emit", 7: "emit",
		}
		self.mouse_mode = 0
		self._right_drag_pos = None

		# ── Shapes ──
		self.shapes = {}
		self.display_shapes = True
		self.active_shape = (None, 1.0, 0.0, 0.0)
		self.eraser_shape = None

		# ── Display state tracking ──
		self._display_states = []
		self.frozen = False
		self.is_excluded = False

		# ── Zoom ──
		self.mag = 1.1
		self.invmag = 1.0 / self.mag

		# ── Drawing params ──
		self.drawr1 = 5
		self.drawv1 = 1.0
		self.drawr2 = 5
		self.drawv2 = 0.0

		# ── Histogram buffer ──
		self.hist = None

		# ── PyGfx setup ──
		self._canvas = None
		self._renderer = None
		self._gfx_texture = None
		self._gfx_image_node = None
		self._scene = gfx.Scene()
		self._shape_group = None
		self._camera = None
		self._dirty = True

		self._setup_gfx()

		# ── Inspector ──
		self.inspector = None

		# Load the data if provided
		if stack is not None:
			self.set_data(stack, metadata=metadata)

	def _setup_gfx(self):
		"""Initialize PyGfx rendering pipeline with rendercanvas Qt backend."""
		try:
			self._canvas = _EMCanvas(parent_widget=self, parent=self)

			layout = QtWidgets.QVBoxLayout(self)
			layout.setContentsMargins(0, 0, 0, 0)
			layout.addWidget(self._canvas)

			self._renderer = gfx.WgpuRenderer(self._canvas)
			self._wgpu_device = self._renderer.device

			self._camera = gfx.OrthographicCamera()
			self._camera.world.position = (0, 0, 1)

			self.setAcceptDrops(True)
			self.setContextMenuPolicy(Qt.PreventContextMenu)

			self._canvas.request_draw(self._render_callback)
		except Exception as e:
			print(f"Warning: PyGfx init failed ({e}), using fallback rendering")
			self._canvas = None
			self._renderer = None
			self._camera = None

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
			print(f"Render error: {e}")

		self._canvas.request_draw(self._render_callback)

	# ── Properties ──

	@property
	def nimg(self):
		if self._list_data is not None:
			return len(self._list_data)
		return 1 if self._data_array is not None else 0

	@property
	def current_index(self):
		return self._list_idx

	def get_data_dims(self):
		if self._data_array is not None:
			return self._data_array.shape
		return (0, 0)

	def get_data(self):
		return self._data_array

	def set_data(self, stack_or_data, file_name="", metadata=None):
		"""Set the data to display. Accepts EMStack2D/3D, numpy array, or list."""
		self._file_name = file_name
		if file_name:
			import os
			self.setWindowTitle(os.path.basename(file_name))

		if self._metadata and not metadata:
			pass
		elif metadata is not None:
			self._metadata = metadata

		incoming = stack_or_data

		if incoming is None:
			self._data_array = None
			self._list_data = None
			return

		if hasattr(incoming, 'numpy'):
			incoming = incoming.numpy()

		if not isinstance(incoming, (list, tuple)):
			arr = np.asarray(incoming)
			if arr.ndim == 3:
				self._list_data = [arr[:, :, i] for i in range(arr.shape[2])]
				self._list_idx = 0
				self._data_array = self._list_data[0]
			elif arr.ndim == 2:
				self._data_array = arr.astype(np.float32)
				self._list_data = None
			else:
				print(f"Warning: unsupported array dimensions {arr.ndim}")
				return
		else:
			self._list_data = [np.asarray(d, dtype=np.float32) for d in incoming]
			self._list_idx = 0
			self._data_array = self._list_data[0]

		# Auto-contrast on new data
		self.auto_contrast(inspector_update=False, display_update=False)

		# Set initial scale to match display/image ratio (native zoom)
		ny, nx = self._data_array.shape
		w = self.width() if self.width() > 0 else 512
		h = self.height() if self.height() > 0 else 512
		scale_w = float(w) / nx
		scale_h = float(h) / ny
		self.scale = min(scale_w, scale_h)
		# Center: origin = screen position of data[0,0] from top-left
		self.origin = ((w - nx * self.scale) / 2.0, (h - ny * self.scale) / 2.0)
		self.invmag = 1.0 / self.mag

		# Update inspector
		if self.inspector:
			self.inspector_update(use_fourier=False)
			if self._list_data is not None:
				self.inspector.enable_image_range(0, len(self._list_data), self._list_idx)
			else:
				self.inspector.disable_image_range()

		self.force_display_update()

	def set_shapes(self, shapes_dict):
		self.shapes = shapes_dict
		self._request_render()

	def update_shapes(self, shapes_dict):
		self.shapes.update(shapes_dict)
		self._request_render()

	def add_shape(self, name, shape):
		self.shapes[name] = shape
		self._request_render()

	def del_shape(self, name):
		self.shapes.pop(name, None)
		self._request_render()

	def del_shapes(self, pattern=""):
		keys = list(self.shapes.keys())
		for k in keys:
			if pattern and pattern.lower() not in k.lower():
				continue
			self.shapes.pop(k, None)
		if not pattern:
			self.shapes.clear()
		self._request_render()

	def set_scale(self, new_scale):
		"""Adjust zoom level, keeping image center stable."""
		if abs(new_scale - self.scale) < 1e-6:
			return
		w = self.width()
		h = self.height()
		# Keep the screen center mapped to the same data point
		old_center_data_x = (w/2.0 - self.origin[0]) / self.scale
		old_center_data_y = (h/2.0 - self.origin[1]) / self.scale
		self.origin = (
			w/2.0 - old_center_data_x * new_scale,
			h/2.0 - old_center_data_y * new_scale,
		)
		self.scale = new_scale
		self.signal_set_scale.emit(new_scale)
		self._request_render()

	def set_origin(self, x, y):
		if self.origin == (x, y):
			return
		self.origin = (x, y)
		self.origin_update.emit((x, y))
		self._request_render()

	def get_origin(self):
		return self.origin

	def set_invert(self, val):
		self.invert = bool(val)
		self._request_render()

	def set_gamma(self, val):
		self.gamma = float(val)
		self._request_render()

	def set_density_range(self, min_d, max_d):
		self.curmin = float(min_d)
		self.curmax = float(max_d)
		self._request_render()

	def set_density_min(self, val):
		self.curmin = float(val)
		self._request_render()

	def density_min(self):
		return self.curmin

	def density_max(self):
		return self.curmax

	def set_mouse_mode(self, mode_num):
		"""Switch mouse interaction mode. 0=normal, 3=probe, 4=measure, 5=draw."""
		if mode_num in self._mouse_modes:
			self.mouse_mode = mode_num
			# Clear probe/measure overlays when switching modes
			self.shapes.pop("PROBE", None)
			self.shapes.pop("MEAS", None)

	def auto_contrast(self, inspector_update=True, display_update=True):
		"""Auto-adjust contrast to mean +/- 3 sigma."""
		data = self._get_current_data()
		if data is None:
			return

		mean = float(np.mean(data))
		sigma = float(np.std(data))
		m0 = float(np.min(data))
		m1 = float(np.max(data))

		self.minden = m0
		self.maxden = m1
		self.curmin = max(m0, mean - 3.0 * sigma)
		self.curmax = min(m1, mean + 3.0 * sigma)

		if inspector_update and self.inspector:
			self.inspector_update(use_fourier=False)
		if display_update:
			self.force_display_update()

	def full_contrast(self, inspector_update=True, display_update=True):
		"""Set contrast to full data range."""
		data = self._get_current_data()
		if data is None:
			return

		self.curmin = float(np.min(data))
		self.curmax = float(np.max(data))
		self.minden = self.curmin
		self.maxden = float(np.max(data))

		if inspector_update and self.inspector:
			self.inspector_update(use_fourier=False)
		if display_update:
			self.force_display_update()

	def auto_contrast_for_current_mode(self, inspector_update=True, display_update=True):
		"""Auto-contrast that respects current display mode (real or FFT)."""
		if self.curfft > 0 and self._display_fft is not None:
			self.auto_contrast_fft(inspector_update, display_update)
		else:
			self.auto_contrast(inspector_update, display_update)

	def full_contrast_for_current_mode(self, inspector_update=True, display_update=True):
		"""Full contrast that respects current display mode (real or FFT)."""
		if self.curfft > 0 and self._display_fft is not None:
			self.full_contrast_fft(inspector_update, display_update)
		else:
			self.full_contrast(inspector_update, display_update)

	def auto_contrast_fft(self, inspector_update=True, display_update=True):
		"""Auto-contrast on current FFT display data."""
		data = self._display_fft
		if data is None:
			return
		mean = float(np.mean(data))
		sigma = float(np.std(data))
		m0 = float(np.min(data))
		m1 = float(np.max(data))
		self.fminden = m0
		self.fmaxden = m1
		self.fcurmin = max(m0, mean - 3.0 * sigma)
		self.fcurmax = min(m1, mean + 3.0 * sigma)
		if inspector_update and self.inspector:
			self.inspector_update(use_fourier=True)
		if display_update:
			self.force_display_update()

	def full_contrast_fft(self, inspector_update=True, display_update=True):
		"""Full contrast on current FFT display data."""
		data = self._display_fft
		if data is None:
			return
		self.fcurmin = float(np.min(data))
		self.fcurmax = float(np.max(data))
		self.fminden = self.fcurmin
		self.fmaxden = self.fcurmax
		if inspector_update and self.inspector:
			self.inspector_update(use_fourier=True)
		if display_update:
			self.force_display_update()

	def set_FFT(self, mode):
		"""Switch FFT display mode. 0=real, 1=complex, 2=amplitude, 3=phase."""
		self.curfft = mode
		self._compute_fft_display(mode)
		self.inspector_update(use_fourier=(mode > 0))
		self.force_display_update()

	def _compute_fft_display(self, mode):
		data = self._get_current_data()
		if data is None:
			return

		cache_key = (data.shape, mode)
		if cache_key in self._fft_cache:
			self._display_fft = self._fft_cache[cache_key]
		else:
			try:
				fft_data = np.fft.fft2(data)
				fft_shifted = np.fft.fftshift(fft_data)

				if mode == 1:
					self._display_fft = fft_shifted
				elif mode == 2:
					amp = np.abs(fft_shifted)
					self._display_fft = amp
				elif mode == 3:
					phase = np.angle(fft_shifted)
					self._display_fft = phase

				self._fft_cache[cache_key] = self._display_fft
			except Exception as e:
				print(f"FFT compute error: {e}")

		# Always recalculate contrast params for this mode (they may have been clobbered)
		if mode in (2, 3):
			m0 = float(np.min(self._display_fft))
			m1 = float(np.max(self._display_fft))
			mean = float(np.mean(self._display_fft))
			sigma = float(np.std(self._display_fft))
			self.fminden = m0
			self.fmaxden = min(m1, mean + 20.0 * sigma)
			self.fcurmin = m0
			self.fcurmax = self.fmaxden
		elif mode == 1:
			m0 = float(np.min(np.abs(self._display_fft)))
			m1 = float(np.max(np.abs(self._display_fft)))
			self.fminden = m0
			self.fmaxden = m1
			self.fcurmin = m0
			self.fcurmax = m1
		else:
			self.fcurmin = 0
			self.fcurmax = 1

	def _get_current_data(self):
		if self._data_array is None:
			return None
		return self._data_array

	def force_display_update(self):
		self._dirty = True
		self._request_render()

	def _request_render(self):
		if self._canvas and self._renderer:
			self._dirty = True

	def _render(self):
		"""Main render pipeline. Converts data to RGB bitmap and updates PyGfx texture."""
		data = self._get_current_data()
		if data is None:
			return

		w = self.width()
		h = self.height()
		if w == 0 or h == 0:
			return

		use_fft = self.curfft in (1, 2, 3)

		if use_fft and self._display_fft is not None:
			if self.curfft == 1:
				rgb, hist = _render_fft_complex_rgb(
					self._display_fft, self.scale, *self.origin, w, h,
					self.fcurmin, self.fcurmax, self.fgamma, self.invert,
				)
			else:
				fft_data = np.asarray(self._display_fft, dtype=np.float32)
				rgb, hist = _render_image_8bit(
					fft_data, self.scale, *self.origin, w, h,
					self.fcurmin, self.fcurmax, self.fgamma, self.invert, 0,
				)
		else:
			rgb, hist = _render_image_8bit(
				data.astype(np.float32), self.scale, *self.origin, w, h,
				self.curmin, self.curmax, self.gamma, self.invert, self.histogram_mode,
			)

		self.hist = hist

		# Update PyGfx texture (float32 in [0..1], no post-processing contrast/brightness)
		self._update_texture(rgb)

		# Render shape overlays on top
		if self.display_shapes and self.shapes:
			self._render_shapes(w, h)

		# Apply frozen/excluded overlays (rgb is now float32 in [0..1])
		if self.frozen or self.is_excluded:
			if self.is_excluded:
				overlay_color = np.array([0.0, 0.1, 0.0])
			else:
				overlay_color = np.array([0.0, 0.0, 0.1])
			alpha = 0.15
			final_rgb = rgb * (1 - alpha) + overlay_color * alpha
			final_rgb = np.clip(final_rgb, 0.0, 1.0)
			self._update_texture(final_rgb)

	# ── Texture management ──

	def _update_texture(self, rgb_float):
		"""Update the display texture with new RGBA float32 data."""
		h, w, c = rgb_float.shape

		if self._gfx_texture is None or self._gfx_texture.size[0] != w or self._gfx_texture.size[1] != h:
			self._create_texture(w, h)

		# PyGfx stores textures as float32 internally
		rgba = np.zeros((h, w, 4), dtype=np.float32)
		rgba[:, :, :3] = rgb_float
		rgba[:, :, 3] = 1.0
		self._gfx_texture.set_data(rgba)

	def _create_texture(self, w, h):
		"""Create the PyGfx texture and image node for display."""
		try:
			empty_data = np.zeros((h, w, 4), dtype=np.float32)
			self._gfx_texture = gfx.Texture(empty_data, dim=2, colorspace='physical')

			geometry = gfx.Geometry(grid=self._gfx_texture)
			material = gfx.ImageBasicMaterial()
			image_node = gfx.Image(geometry=geometry, material=material)
			image_node.world_bounds = ((0, 0, 0), (w, h, 0))

			if self._gfx_image_node is not None:
				self._scene.remove(self._gfx_image_node)
			self._gfx_image_node = image_node
			self._scene.add(image_node)

		except Exception as e:
			print(f"Texture creation failed: {e}")

	# ── Shape rendering ──

	def _render_shapes(self, w, h):
		"""Render shape overlays on top of the image."""
		if self._shape_group is None:
			self._shape_group = gfx.Group()
			self._shape_group.render_order = 10
			self._scene.add(self._shape_group)

		# Clear old shapes
		while len(self._shape_group.children) > 0:
			self._shape_group.remove(self._shape_group.children[0])

		for name, shape in self.shapes.items():
			node = self._render_single_shape(shape, w, h)
			if node:
				self._shape_group.add(node)

	def _render_single_shape(self, shape, w, h):
		"""Render a single shape as a PyGfx node."""
		try:
			if isinstance(shape, ShapeLine):
				x0s, y0s = self._img_to_scr(shape.x0, shape.y0, w, h)
				x1s, y1s = self._img_to_scr(shape.x1, shape.y1, w, h)
				return self._make_line_node(
					x0s, y0s, x1s, y1s, shape.r, shape.g, shape.b, True)

			elif isinstance(shape, ShapeRCircle):
				# Data-space inscribed circle from bounding rectangle
				sx0, sy0 = self._img_to_scr(shape.x0, shape.y0, w, h)
				sx1, sy1 = self._img_to_scr(shape.x1, shape.y1, w, h)
				sx_c = (sx0 + sx1) / 2.0
				sy_c = (sy0 + sy1) / 2.0
				sr = min(abs(sx1 - sx0), abs(sy1 - sy0)) / 2.0
				return self._make_circle_node(sx_c, sy_c, sr, shape.r, shape.g, shape.b, False)

			elif isinstance(shape, ShapeCircle):
				# Data-space circle
				sx, sy = self._img_to_scr(shape.cx, shape.cy, w, h)
				sr = shape.radius * self.scale
				return self._make_circle_node(sx, sy, sr, shape.r, shape.g, shape.b, True)

			elif isinstance(shape, ShapeScrLabel):
				# Screen-space label
				return self._make_text_node(
					shape.text, shape.x, shape.y,
					shape.r, shape.g, shape.b, True)

			elif isinstance(shape, ShapeLabel):
				# Data-space label
				return self._make_text_node(
					shape.text, shape.x, shape.y,
					shape.r, shape.g, shape.b, False)

			elif isinstance(shape, ShapeRectPoint):
				# Data-space rectangle with point marker (used for probe overlay)
				x0s, y0s = self._img_to_scr(shape.x0, shape.y0, w, h)
				x1s, y1s = self._img_to_scr(shape.x1, shape.y1, w, h)
				return self._make_rect_node(x0s, y0s, x1s, y1s, shape.r, shape.g, shape.b, True)

		except Exception:
			pass
		return None

	def _make_line_node(self, x0, y0, x1, y1, r, g, b, flip_y):
		if flip_y:
			y0 = self.height() - y0
			y1 = self.height() - y1
		pts = np.array([[x0, y0, 0], [x1, y1, 0]], dtype=np.float32)
		return gfx.Line(
			gfx.Geometry(positions=pts),
			gfx.LineBasicMaterial(color=(r, g, b), linewidth=2))

	def _make_rect_node(self, x0, y0, x1, y1, r, g, b, flip_y):
		"""Create a PyGfx rectangle (line strip around corners)."""
		if flip_y:
			y0 = self.height() - y0
			y1 = self.height() - y1
		pts = np.array([
			[x0, y0, 0], [x1, y0, 0], [x1, y1, 0],
			[x0, y1, 0], [x0, y0, 0],
		], dtype=np.float32)
		return gfx.Line(
			gfx.Geometry(positions=pts),
			gfx.LineBasicMaterial(color=(r, g, b), linewidth=2))

	def _make_circle_node(self, cx, cy, radius, r, g, b, flip_y):
		if flip_y:
			cy = self.height() - cy
		n_points = 64
		thetas = np.linspace(0, 2 * np.pi, n_points)
		pts = np.column_stack([
			cx + radius * np.cos(thetas),
			cy + radius * np.sin(thetas),
			np.zeros(n_points)
		]).astype(np.float32)
		return gfx.Line(
			gfx.Geometry(positions=pts),
			gfx.LineBasicMaterial(color=(r, g, b), linewidth=2))

	def _make_text_node(self, text, x, y, r, g, b, is_screen_space):
		if not is_screen_space:
			x, y = self._img_to_scr(x, y, self.width(), self.height())
		return gfx.Text(
			text, position=(x, y, 0), color=(r, g, b, 1.0),
			render_order=20)

	def _img_to_screen_pts(self, pts_img, w, h):
		result = []
		for ix, iy in pts_img:
			sx = self.origin[0] + ix * self.scale
			sy = self.origin[1] + iy * self.scale
			result.append((sx, sy))
		return result

	def _img_to_scr(self, ix, iy, w, h):
		sx = self.origin[0] + ix * self.scale
		sy = self.origin[1] + iy * self.scale
		return sx, sy

	def _scr_to_img(self, sx, sy):
		ix = int((sx - self.origin[0]) / self.scale)
		iy = int((sy - self.origin[1]) / self.scale)
		return ix, iy

	def set_file_name(self, name):
		self._file_name = name

	def get_file_name(self):
		return self._file_name

	# ── Drawing helpers ──

	def add_vector_overlay(self, x0, y0, x1, y1, v=None):
		"""Add vector field as line overlays."""
		n = len(x0)
		shapes_dict = {}
		for i in range(n):
			if v is None:
				color = (0.2, 0.8, 0.2)
			else:
				vn = (v[i] - min(v)) / (max(v) - min(v)) if max(v) > min(v) else 0.5
				color = (1.0 - vn, 0.2, vn)
			shapes_dict[f"vl_{i}"] = ShapeLine(*color, x0[i], y0[i], x1[i], y1[i])
		self.shapes.update(shapes_dict)
		self._request_render()

	# ── Mouse handling (forwarded from _EMCanvas) ──

	def _on_mouse_press(self, event):
		self._handle_mouse_press(event)

	def _on_mouse_move(self, event):
		self._handle_mouse_move(event)

	def _on_mouse_release(self, event):
		self._handle_mouse_release(event)

	def _on_wheel(self, event):
		self._handle_wheel(event)

	def _on_key_press(self, event):
		self._handle_key_press(event)

	# ── Internal handlers ──

	def _handle_mouse_press(self, event):
		# Middle-click or Alt+Left toggles inspector control panel
		if event.button() == Qt.MouseButton.MiddleButton or (
			event.button() == Qt.MouseButton.LeftButton and
			event.modifiers() & Qt.AltModifier
		):
			if self.inspector and self.inspector.isVisible():
				self.show_inspector(False)
			else:
				self.show_inspector(True)
			return

		# Right-click or Ctrl+Shift+Left starts panning
		if event.button() == Qt.MouseButton.RightButton or (
			event.button() == Qt.MouseButton.LeftButton and
			event.modifiers() & Qt.ControlModifier and
			event.modifiers() & Qt.ShiftModifier
		):
			self._canvas.setCursor(QtCore.Qt.ClosedHandCursor)
			self._right_drag_pos = (event.position().x(), event.position().y())
			return

		lc = self._scr_to_img(event.position().x(), event.position().y())
		mode = self._mouse_modes.get(self.mouse_mode, "emit")

		if mode == "emit":
			self.mousedown.emit(event, lc)
		elif mode == "probe":
			if event.button() == Qt.MouseButton.LeftButton:
				self._do_probe(lc[0], lc[1])
		elif mode == "measure":
			if event.button() == Qt.MouseButton.LeftButton:
				self.shapes.pop("MEAS", None)
				self.shapes["MEAS"] = ShapeLine(0.5, 0.1, 0.5, lc[0], lc[1], lc[0]+1, lc[1])
				self._request_render()
		elif mode == "draw":
			if event.button() == Qt.MouseButton.LeftButton:
				pass

	def _handle_mouse_move(self, event):
		lc = self._scr_to_img(event.position().x(), event.position().y())

		if self._right_drag_pos:
			# Right-drag: pan image. Negate dx/dy so dragging moves image in that direction.
			dx = self._right_drag_pos[0] - event.position().x()
			dy = self._right_drag_pos[1] - event.position().y()
			self.set_origin(self.origin[0] - dx, self.origin[1] + dy)
			self._right_drag_pos = (event.position().x(), event.position().y())
			return

		mode = self._mouse_modes.get(self.mouse_mode, "emit")

		if mode == "emit":
			if event.buttons() & Qt.MouseButton.LeftButton:
				self.mousedrag.emit(event, lc)
			else:
				self.mousemove.emit(event, lc)
		elif mode == "probe":
			if event.buttons() & Qt.MouseButton.LeftButton:
				self._do_probe(lc[0], lc[1])
		elif mode == "measure":
			if event.buttons() & Qt.MouseButton.LeftButton:
				self._update_measure_shape(lc)

	def _handle_mouse_release(self, event):
		self._canvas.unsetCursor()
		if self._right_drag_pos:
			self._right_drag_pos = None
			return

		lc = self._scr_to_img(event.position().x(), event.position().y())
		mode = self._mouse_modes.get(self.mouse_mode, "emit")

		if mode == "emit":
			self.mouseup.emit(event, lc)

	def _handle_wheel(self, event):
		if self.mouse_mode == 0 and event.modifiers() & Qt.ShiftModifier:
			event.ignore()
			return

		delta = event.angleDelta().y()
		if delta > 0:
			self.set_scale(self.scale * self.mag)
		elif delta < 0:
			self.set_scale(self.scale * self.invmag)

		if self.inspector:
			self.inspector.set_scale(self.scale)

	def _handle_key_press(self, event):
		if event.key() == Qt.Key_Up:
			self.increment_list_data(1)
		elif event.key() == Qt.Key_Down:
			self.increment_list_data(-1)
		elif event.key() == Qt.Key_Space:
			self.display_shapes = not self.display_shapes
			self._request_render()
		elif event.key() == Qt.Key_F:
			new_mode = self.curfft ^ 2
			if new_mode > 3:
				new_mode = min(new_mode, 3)
			self.set_FFT(new_mode)
		elif event.key() == Qt.Key_C:
			self.auto_contrast()
		elif event.key() == Qt.Key_T:
			self.full_contrast()
		elif event.key() in (Qt.Key_Left, Qt.Key_Right):
			pass

	# ── Inspector integration ──

	def inspector_update(self, use_fourier=False):
		if self.inspector:
			if not use_fourier:
				self.inspector.set_limits(self.minden, self.maxden, self.curmin, self.curmax)
				self.inspector.set_gamma(self.gamma)
			else:
				self.inspector.set_limits(self.fminden, self.fmaxden, self.fcurmin, self.fcurmax)
				self.inspector.set_gamma(self.fgamma)
			self.inspector._update_fft_buttons(self.curfft)
			self.inspector.set_scale(self.scale)

	def get_inspector(self):
		if not self.inspector:
			self.inspector = EMImageInspector2D(self)
			self.inspector_update()
			if self._list_data is not None:
				self.inspector.enable_image_range(0, len(self._list_data), self._list_idx)
			self.inspector._update_brightness_contrast()
		return self.inspector

	def show_inspector(self, show):
		"""Show or hide the inspector control panel."""
		if show:
			insp = self.get_inspector()
			insp.show()
			insp.raise_()
			insp.activateWindow()
		elif self.inspector:
			self.inspector.hide()

	def coords_within_image_bounds(self, x, y):
		dims = self.get_data_dims()
		if dims == (0, 0):
			return False
		return 0 <= x < dims[0] and 0 <= y < dims[1]

	def increment_list_data(self, delta):
		if self._list_data is None:
			return

		new_idx = self._list_idx + delta
		if 0 <= new_idx < len(self._list_data):
			self._list_idx = new_idx
			self._data_array = self._list_data[self._list_idx]
			self._fft_cache.clear()
			if self.curfft > 0:
				self._compute_fft_display(self.curfft)

			if self.inspector:
				self.inspector.set_image_idx(self._list_idx, 1)

			self.force_display_update()
			self.signal_increment_list_data.emit(delta)

	def _update_measure_shape(self, lc):
		if "MEAS" in self.shapes:
			shape = self.shapes["MEAS"]
			shape.x1 = lc[0]
			shape.y1 = lc[1]
			self._request_render()

	def _do_probe(self, x, y):
		"""Probe data values at (x, y) with area statistics."""
		data = self._get_current_data()
		if data is None:
			return

		ny, nx = data.shape
		try:
			sz = int(self.inspector.ptareasize.getValue()) if self.inspector else 32
		except Exception:
			sz = 32
		if sz < 1:
			sz = 1

		x, y = int(round(x)), int(round(y))

		# Clamp to image bounds
		half_sz = sz // 2
		area_x0 = max(0, min(x - half_sz, nx))
		area_y0 = max(0, min(y - half_sz, ny))
		area_x1 = max(0, min(x + half_sz + (sz % 2), nx))
		area_y1 = max(0, min(y + half_sz + (sz % 2), ny))

		# If probe area has no overlap with image, abort gracefully
		if area_x1 <= area_x0 or area_y1 <= area_y0:
			self.shapes.pop("PROBE", None)
			if self.inspector:
				self.inspector.set_probe_values(None, None, None, None, None, None, None, x, y, nx, ny)
			return

		# Center point value
		if 0 <= x < nx and 0 <= y < ny:
			center_val = float(data[y, x])
		else:
			center_val = None

		# Area statistics from the clipped region
		area_data = data[area_y0:area_y1, area_x0:area_x1]
		area_avg = float(np.mean(area_data))
		area_sig = float(np.std(area_data))

		# Non-zero statistics (excluding zeros)
		nz_mask = area_data != 0.0
		if np.any(nz_mask):
			nz_data = area_data[nz_mask]
			nz_avg = float(np.mean(nz_data))
			nz_sig = float(np.std(nz_data))
		else:
			nz_avg = 0.0
			nz_sig = 0.0

		# Skewness and kurtosis (Fisher definitions)
		n_pts = area_data.size
		if n_pts > 2:
			centered = area_data - area_avg
			m2 = np.mean(centered**2)
			m3 = np.mean(centered**3)
			m4 = np.mean(centered**4)
			if m2 > 1e-30:
				skew = m3 / (m2 ** 1.5)
				kurt = m4 / (m2 ** 2) - 3.0
			else:
				skew = 0.0
				kurt = 0.0
		else:
			skew = 0.0
			kurt = 0.0

		# Update inspector labels
		if self.inspector:
			self.inspector.set_probe_values(
				center_val, area_avg, nz_avg,
				area_sig, nz_sig, skew, kurt,
				x, y, nx, ny)

		# Draw probe rectangle overlay (in data coordinates)
		self.shapes.pop("PROBE", None)
		self.add_shape("PROBE", ShapeRectPoint(0.5, 0.5, 0.1, float(area_x0), float(area_y0), float(area_x1), float(area_y1), 2))
		self._request_render()

	def closeEvent(self, event):
		try:
			if self.inspector:
				self.inspector.close()
		except Exception:
			pass
		super().closeEvent(event)

	def resizeEvent(self, event):
		w = self.width()
		h = self.height()
		if w > 0 and h > 0:
			self._handle_resize(w, h)

	def _handle_resize(self, width, height):
		data = self._get_current_data()
		if data is None:
			return

		ny, nx = data.shape
		# Center the image: origin = screen position of data[0,0] from top-left
		self.origin = (
			(width - nx * self.scale) / 2.0,
			(height - ny * self.scale) / 2.0,
		)

		if self._camera is not None:
			self._camera.width = width
			self._camera.height = height

		if self._gfx_image_node is not None:
			self._gfx_image_node.world_bounds = ((0, 0, 0), (width, height, 0))

		self._request_render()

	def leaveEvent(self, event):
		if self._right_drag_pos:
			self._right_drag_pos = None
		if self._canvas:
			self._canvas.unsetCursor()


# ─── Inspector / Control Panel ──────────────────────────────────────────

class EMImageInspector2D(QtWidgets.QWidget):
	"""Control panel for the 2D image viewer."""

	def __init__(self, target):
		super().__init__()
		self._target_ref = weakref.ref(target)
		self._busy_slider = 0

		self.setWindowTitle("Image Controls")
		self.setWindowFlags(Qt.Window)

		self.vbl = QtWidgets.QVBoxLayout(self)
		self.vbl.setContentsMargins(4, 4, 4, 4)
		self.vbl.setSpacing(4)

		self._build_tabs()
		self._build_contrast_controls()

		self._wire_signals()

	def target(self):
		if self._target_ref:
			return self._target_ref()
		return None

	def _wire_signals(self):
		"""Connect slider valueChanged signals to widget parameter setters."""
		self.min_slider.valueChanged.connect(self._on_min_changed)
		self.max_slider.valueChanged.connect(self._on_max_changed)
		self.gamma_slider.valueChanged.connect(self._on_gamma_changed)
		self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
		self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
		self.scale_slider.valueChanged.connect(self._on_scale_changed)
		self.n_slider.valueChanged.connect(self._on_n_changed)

	def _on_min_changed(self, val):
		if self._busy_slider:
			return
		self._busy_slider = 1
		tgt = self.target()
		if tgt:
			tgt.set_density_min(val)
		self._update_brightness_contrast()
		self._busy_slider = 0

	def _on_max_changed(self, val):
		if self._busy_slider:
			return
		self._busy_slider = 1
		tgt = self.target()
		if tgt:
			tgt.set_density_range(self._get_current_min(), val)
		self._update_brightness_contrast()
		self._busy_slider = 0

	def _on_gamma_changed(self, val):
		tgt = self.target()
		if tgt:
			tgt.set_gamma(val)

	def _on_brightness_changed(self, val):
		if self._busy_slider:
			return
		self._busy_slider = 1
		self._update_min_max()
		self._busy_slider = 0

	def _on_contrast_changed(self, val):
		if self._busy_slider:
			return
		self._busy_slider = 1
		self._update_min_max()
		self._busy_slider = 0

	def _on_scale_changed(self, val):
		tgt = self.target()
		if tgt:
			tgt.set_scale(float(val))

	def _on_n_changed(self, val):
		"""Navigate to a different image in the stack."""
		if self._busy_slider:
			return
		self._busy_slider = 1
		tgt = self.target()
		if tgt and hasattr(tgt, '_list_data') and tgt._list_data is not None:
			new_idx = int(val)
			if new_idx != tgt._list_idx:
				tgt._list_idx = new_idx
				tgt._data_array = tgt._list_data[new_idx]
				tgt._fft_cache.clear()
				if tgt.curfft > 0:
					tgt._compute_fft_display(tgt.curfft)
				tgt.auto_contrast(inspector_update=False, display_update=False)
				if tgt.inspector:
					tgt.inspector_update(use_fourier=False)
					tgt.inspector.set_image_idx(new_idx)
				tgt.force_display_update()
		self._busy_slider = 0

	def _get_current_min(self):
		tgt = self.target()
		return tgt.curmin if tgt else 0.0

	def _update_brightness_contrast(self):
		"""Update brightness/contrast sliders to reflect current min/max values."""
		tgt = self.target()
		if not tgt:
			return
		lowlim = tgt.minden
		highlim = tgt.maxden
		min_val = self.min_slider.getValue()
		max_val = self.max_slider.getValue()

		range_diff = highlim - lowlim
		if abs(range_diff) < 1e-12:
			return

		b = 0.5 * (min_val + max_val - (lowlim + highlim)) / range_diff
		c = (min_val - max_val) / (2.0 * (lowlim - highlim))
		brts = -b
		conts = 1.0 - c

		self.brightness_slider.setValue(brts, quiet=1)
		self.contrast_slider.setValue(conts, quiet=1)

	def _update_min_max(self):
		"""Update min/max sliders to reflect current brightness/contrast values."""
		tgt = self.target()
		if not tgt:
			return
		lowlim = tgt.minden
		highlim = tgt.maxden
		brt_val = self.brightness_slider.getValue()
		cont_val = self.contrast_slider.getValue()

		range_diff = highlim - lowlim
		center = (lowlim + highlim) / 2.0

		x0 = center - range_diff * (1.0 - cont_val) - brt_val * range_diff
		x1 = center + range_diff * (1.0 - cont_val) - brt_val * range_diff

		self.min_slider.setValue(x0, quiet=1)
		self.max_slider.setValue(x1, quiet=1)
		tgt.set_density_range(x0, x1)

	def _build_tabs(self):
		"""Build the inspector tab widget with mouse mode switching.

		Tab indices: 0=App, 1=Save, 2=Filt, 3=Probe, 4=Meas, 5=Draw, 6=PSpec
		Mouse modes: emit, emit, emit, probe, measure, draw, emit
		"""
		tab_widget = QtWidgets.QTabWidget()

		tab_widget.addTab(QtWidgets.QTextEdit("Application mouse functions"), "App")
		tab_widget.addTab(self._build_save_tab(), "Save")
		tab_widget.addTab(self._build_filter_tab(), "Filt")
		tab_widget.addTab(self._build_probe_tab(), "Probe")
		tab_widget.addTab(self._build_measure_tab(), "Meas")
		tab_widget.addTab(self._build_draw_tab(), "Draw")
		tab_widget.addTab(self._build_pspec_tab(), "PSpec")

		# Wire tab changes to switch mouse modes
		tab_map = {0: 0, 1: 0, 2: 0, 3: 3, 4: 4, 5: 5, 6: 0}
		tab_widget.currentChanged.connect(lambda idx: self._set_mouse_mode(tab_map.get(idx, 0)))

		self.vbl.addWidget(tab_widget)

	def _build_save_tab(self):
		tab = QtWidgets.QWidget()
		layout = QtWidgets.QGridLayout(tab)

		snap_btn = QtWidgets.QPushButton("Snapshot")
		snap_btn.clicked.connect(self.do_snapshot)
		save_btn = QtWidgets.QPushButton("Save Img")
		save_btn.clicked.connect(self.do_saveimg)
		stack_btn = QtWidgets.QPushButton("Save Stack")
		stack_btn.clicked.connect(self.do_savestack)

		layout.addWidget(snap_btn, 0, 0)
		layout.addWidget(save_btn, 1, 0)
		layout.addWidget(stack_btn, 1, 1)

		return tab

	def _build_filter_tab(self):
		tab = QtWidgets.QWidget()
		layout = QtWidgets.QGridLayout(tab)

		self.procbox1 = StringBox(label="Process1:", value="filter.lowpass.gauss:cutoff_abs=0.125")
		self.procbox2 = StringBox(label="Process2:", value="filter.highpass.gauss:cutoff_pixels=3")
		self.procbox3 = StringBox(label="Process3:", value="math.linear:scale=5:shift=0")

		layout.addWidget(self.procbox1, 0, 0)
		layout.addWidget(self.procbox2, 1, 0)
		layout.addWidget(self.procbox3, 2, 0)

		label = QtWidgets.QLabel("Display only - image unchanged!")
		layout.addWidget(label, 3, 0)

		return tab

	def _build_probe_tab(self):
		tab = QtWidgets.QWidget()
		layout = QtWidgets.QGridLayout(tab)

		self.ptareasize = ValBox(label="Probe Size:", value=32)
		self.ptareasize.setIntonly(True)
		layout.addWidget(self.ptareasize, 0, 0, 1, 2)

		self.ptpointval = QtWidgets.QLabel("Point Value: ")
		layout.addWidget(self.ptpointval, 1, 0, 1, 2, Qt.AlignLeft)

		self.ptareaavg = QtWidgets.QLabel("Area Avg: ")
		layout.addWidget(self.ptareaavg, 2, 0)
		self.ptareaavgnz = QtWidgets.QLabel("Area Avg (!=0): ")
		layout.addWidget(self.ptareaavgnz, 2, 1)

		self.ptareasig = QtWidgets.QLabel("Area Sig: ")
		layout.addWidget(self.ptareasig, 3, 0)
		self.ptareasignz = QtWidgets.QLabel("Area Sig (!=0): ")
		layout.addWidget(self.ptareasignz, 3, 1)

		self.ptareaskew = QtWidgets.QLabel("Skewness: ")
		layout.addWidget(self.ptareaskew, 4, 0)
		self.ptcoord = QtWidgets.QLabel("Center Coord: ")
		layout.addWidget(self.ptcoord, 4, 1)

		self.ptareakurt = QtWidgets.QLabel("Kurtosis: ")
		layout.addWidget(self.ptareakurt, 5, 0)
		self.ptcoord2 = QtWidgets.QLabel("")
		layout.addWidget(self.ptcoord2, 5, 1)

		return tab

	def _build_measure_tab(self):
		tab = QtWidgets.QWidget()
		layout = QtWidgets.QGridLayout(tab)

		self.mtapix = ValSlider(label="A/Pix")
		self.mtapix.setRange(0.5, 10.0)
		self.mtapix.setValue(1.0)
		layout.addWidget(self.mtapix, 0, 0, 1, 2)

		self.mtshoworigin = QtWidgets.QLabel("Origin: 0,0")
		layout.addWidget(self.mtshoworigin, 1, 0)
		self.mtshowend = QtWidgets.QLabel("End: 0,0")
		layout.addWidget(self.mtshowend, 1, 1)

		self.mtshowlen = QtWidgets.QLabel("dx,dy: 0")
		layout.addWidget(self.mtshowlen, 2, 0)
		self.mtshowlen2 = QtWidgets.QLabel("Length: 0")
		layout.addWidget(self.mtshowlen2, 2, 1)

		self.mtshowval = QtWidgets.QLabel("Value: ?")
		layout.addWidget(self.mtshowval, 3, 0, 1, 2, Qt.AlignLeft)
		self.mtshowval2 = QtWidgets.QLabel("")
		layout.addWidget(self.mtshowval2, 4, 0, 1, 2, Qt.AlignLeft)

		return tab

	def _build_draw_tab(self):
		tab = QtWidgets.QWidget()
		layout = QtWidgets.QGridLayout(tab)

		layout.addWidget(QtWidgets.QLabel("Pen Size:"), 0, 0)
		self.dtpen = QtWidgets.QLineEdit("5")
		layout.addWidget(self.dtpen, 0, 1)

		layout.addWidget(QtWidgets.QLabel("Pen Val:"), 1, 0)
		self.dtpenv = QtWidgets.QLineEdit("1.0")
		layout.addWidget(self.dtpenv, 1, 1)

		layout.addWidget(QtWidgets.QLabel("Pen Size2:"), 0, 2)
		self.dtpen2 = QtWidgets.QLineEdit("5")
		layout.addWidget(self.dtpen2, 0, 3)

		layout.addWidget(QtWidgets.QLabel("Pen Val2:"), 1, 2)
		self.dtpenv2 = QtWidgets.QLineEdit("0")
		layout.addWidget(self.dtpenv2, 1, 3)

		return tab

	def _build_pspec_tab(self):
		tab = QtWidgets.QWidget()
		layout = QtWidgets.QGridLayout(tab)

		sing_btn = QtWidgets.QPushButton("Single")
		sing_btn.clicked.connect(lambda: self.do_pspec_single(0))
		layout.addWidget(sing_btn, 0, 0)

		az_btn = QtWidgets.QPushButton("Azimuthal")
		az_btn.clicked.connect(lambda: self.do_pspec_az())
		layout.addWidget(az_btn, 1, 0)

		stack_btn = QtWidgets.QPushButton("Stack")
		stack_btn.clicked.connect(self.do_pspec_stack)
		layout.addWidget(stack_btn, 0, 1)

		return tab

	def _build_contrast_controls(self):
		"""Build the button area + sliders that go below the tab widget."""
		tgt = self.target()

		# Button grid next to histogram (Invert, HistEqual, AutoC/FullC, FFT toggles)
		self.hbl = QtWidgets.QHBoxLayout()
		self.hbl.setContentsMargins(0, 0, 0, 0)
		self.hbl.setSpacing(6)

		self.hist_widget = QtWidgets.QLabel("Histogram")
		self.hist_widget.setMinimumSize(128, 64)
		self.hbl.addWidget(self.hist_widget)

		self.btn_grid = QtWidgets.QGridLayout()
		self.btn_grid.setContentsMargins(0, 0, 0, 0)
		self.btn_grid.setSpacing(6)
		self.hbl.addLayout(self.btn_grid)

		# Invert toggle button (row 0)
		self.invert_btn = QtWidgets.QPushButton("Invert")
		self.invert_btn.setCheckable(True)
		self.invert_btn.toggled.connect(lambda v: self._on_invert_toggled(v))
		self.btn_grid.addWidget(self.invert_btn, 0, 0, 1, 1)

		# Histogram equalizer combo (row 0)
		self.hist_equal_combo = QtWidgets.QComboBox()
		self.hist_equal_combo.addItem("Normal")
		self.hist_equal_combo.addItem("Hist Flat")
		self.hist_equal_combo.addItem("Hist Gauss")
		self.hist_equal_combo.currentIndexChanged.connect(self._on_hist_equal_changed)
		self.btn_grid.addWidget(self.hist_equal_combo, 0, 1, 1, 1)

		# Auto Contrast / Full Contrast buttons (row 1)
		self.auto_contrast_btn = QtWidgets.QPushButton("AutoC")
		self.auto_contrast_btn.clicked.connect(self._on_auto_contrast)
		self.btn_grid.addWidget(self.auto_contrast_btn, 1, 0, 1, 1)

		self.full_contrast_btn = QtWidgets.QPushButton("FullC")
		self.full_contrast_btn.clicked.connect(self._on_full_contrast)
		self.btn_grid.addWidget(self.full_contrast_btn, 1, 1, 1, 1)

		# FFT mode toggle buttons (rows 2-3)
		self.fft_group = QtWidgets.QButtonGroup()
		self.fft_group.setExclusive(True)

		self.fft_real_btn = QtWidgets.QPushButton("Real")
		self.fft_real_btn.setCheckable(True)
		self.fft_real_btn.setChecked(True)
		self.btn_grid.addWidget(self.fft_real_btn, 2, 0)
		self.fft_group.addButton(self.fft_real_btn, 0)

		self.fft_fft_btn = QtWidgets.QPushButton("FFT")
		self.fft_fft_btn.setCheckable(True)
		self.btn_grid.addWidget(self.fft_fft_btn, 2, 1)
		self.fft_group.addButton(self.fft_fft_btn, 1)

		self.fft_amp_btn = QtWidgets.QPushButton("Amp")
		self.fft_amp_btn.setCheckable(True)
		self.btn_grid.addWidget(self.fft_amp_btn, 3, 0)
		self.fft_group.addButton(self.fft_amp_btn, 2)

		self.fft_pha_btn = QtWidgets.QPushButton("Pha")
		self.fft_pha_btn.setCheckable(True)
		self.btn_grid.addWidget(self.fft_pha_btn, 3, 1)
		self.fft_group.addButton(self.fft_pha_btn, 3)

		self.fft_group.buttonClicked.connect(self._on_fft_mode_changed)

		self.vbl.addLayout(self.hbl)

		# Vertical sliders in original order: Mag, Min, Max, Brt, Cont, Gam
		self.scale_slider = ValSlider(label="Mag", value=1.0)
		self.scale_slider.setRange(0.1, 20.0)
		if tgt:
			self.scale_slider.setValue(tgt.scale, quiet=1)
		self.vbl.addWidget(self.scale_slider)

		self.min_slider = ValSlider(label="Min", value=0.0)
		if tgt:
			self.min_slider.setValue(tgt.minden, quiet=1)
		self.vbl.addWidget(self.min_slider)

		self.max_slider = ValSlider(label="Max", value=1.0)
		if tgt:
			self.max_slider.setValue(tgt.maxden, quiet=1)
		self.vbl.addWidget(self.max_slider)

		self.brightness_slider = ValSlider(label="Brt", value=0.0)
		self.brightness_slider.setRange(-2.0, 2.0)
		self.vbl.addWidget(self.brightness_slider)

		self.contrast_slider = ValSlider(label="Cont", value=1.0)
		self.contrast_slider.setRange(0.0, 5.0)
		self.vbl.addWidget(self.contrast_slider)

		self.gamma_slider = ValSlider(label="Gam", value=1.0)
		self.gamma_slider.setRange(0.1, 5.0)
		if tgt:
			self.gamma_slider.setValue(tgt.gamma, quiet=1)
		self.vbl.addWidget(self.gamma_slider)

		# Stack image index slider (N)
		self.n_slider = ValSlider(label="N", value=0)
		self.n_slider.setIntonly(True)
		self.n_slider.setEnabled(False)
		self.vbl.addWidget(self.n_slider)

	def _on_invert_toggled(self, checked):
		tgt = self.target()
		if tgt:
			tgt.set_invert(checked)

	def _on_hist_equal_changed(self, idx):
		tgt = self.target()
		if tgt:
			tgt.histogram_mode = idx

	def _on_auto_contrast(self):
		tgt = self.target()
		if tgt:
			tgt.auto_contrast_for_current_mode(inspector_update=True)

	def _on_full_contrast(self):
		tgt = self.target()
		if tgt:
			tgt.full_contrast_for_current_mode(inspector_update=True)

	def _on_fft_mode_changed(self, button):
		tgt = self.target()
		if tgt:
			mode = self.fft_group.id(button)
			tgt.set_FFT(mode)

	def set_limits(self, min_den, max_den, cur_min, cur_max):
		try:
			self.min_slider.setRange(min_den, max_den)
			self.max_slider.setRange(min_den, max_den)
			self.min_slider.setValue(cur_min, quiet=1)
			self.max_slider.setValue(cur_max, quiet=1)
			# Keep brightness/contrast in sync with the new min/max
			self._busy_slider = 1
			self._update_brightness_contrast()
			self._busy_slider = 0
		except Exception:
			pass

	def set_gamma(self, val):
		try:
			self.gamma_slider.setValue(val)
		except Exception:
			pass

	def _update_fft_buttons(self, mode):
		"""Update FFT toggle buttons to reflect current mode."""
		buttons = [self.fft_real_btn, self.fft_fft_btn, self.fft_amp_btn, self.fft_pha_btn]
		for i, btn in enumerate(buttons):
			btn.setChecked(i == mode)

	def set_scale(self, val):
		try:
			self.scale_slider.setValue(val, quiet=1)
		except Exception:
			pass

	def set_image_idx(self, idx, quiet=0):
		"""Set the N slider to a specific image index."""
		self.n_slider.setValue(idx, quiet=quiet)

	def enable_image_range(self, min_val, max_val, current):
		"""Enable stack navigation with N slider."""
		self.n_slider.setEnabled(True)
		self.n_slider.setRange(min_val, max_val - 1)
		self.n_slider.setValue(current, quiet=1)

	def disable_image_range(self):
		"""Disable stack navigation."""
		self.n_slider.setEnabled(False)

	def _set_mouse_mode(self, mode_num):
		"""Switch the widget's mouse mode."""
		tgt = self.target()
		if tgt:
			tgt.mouse_mode = mode_num

	def set_probe_values(self, point_val, area_avg, area_avg_nz,
				area_sig, area_sig_nz, skew, kurt, x, y, nx, ny):
		try:
			self.ptpointval.setText(f"Point Value: {point_val:.3f}")
			self.ptareaavg.setText(f"Area Avg: {area_avg:.3f}")
			self.ptareaavgnz.setText(f"Area Avg (!=0): {area_avg_nz:.3f}")
			self.ptareasig.setText(f"Area Sig: {area_sig:.3f}")
			self.ptareasignz.setText(f"Area Sig (!=0): {area_sig_nz:.3f}")
			self.ptareaskew.setText(f"Skewness: {skew:.3f}")
			self.ptareakurt.setText(f"Kurtosis: {kurt:.3f}")
			self.ptcoord.setText(f"Center Coord: {x}, {y}")
			self.ptcoord2.setText(f"dcen ({x-nx//2}, {y-ny//2})")
		except Exception:
			pass

	def set_measure_info(self, x0, y0, x1, y1, dx, dy, length, apix):
		try:
			self.mtshoworigin.setText(f"Start: {int(x0)}, {int(y0)}")
			self.mtshowend.setText(f"  End: {int(x1)}, {int(y1)}")
			self.mtshowlen.setText(f"dx,dy: {dx:.2f} A, {dy:.2f} A")
			self.mtshowlen2.setText(f"Len: {length:.3f} A")
		except Exception:
			pass

	def do_pspec_single(self, idx):
		pass

	def do_pspec_az(self):
		pass

	def do_pspec_stack(self):
		pass

	def do_snapshot(self):
		tgt = self.target()
		if tgt is None:
			return
		data = tgt.get_data()
		if data is None:
			return
		from PySide6.QtWidgets import QFileDialog
		path, _ = QFileDialog.getSaveFileName(self, "Save Snapshot", "", "Images (*.png *.jpg)")
		if path:
			try:
				from PIL import Image as PILImage
				arr = (np.clip(data, 0, 1) * 255).astype(np.uint8)
				img = PILImage.fromarray(arr)
				img.save(path)
			except ImportError:
				print("PIL required for snapshots")

	def do_saveimg(self):
		pass

	def do_savestack(self):
		pass


def main():
	import sys
	from PySide6.QtWidgets import QApplication

	app = QApplication(sys.argv)

	data = np.random.randn(256, 256).astype(np.float32)
	widget = EMImage2DWidget(stack=data)
	widget.setWindowTitle("EMAN3 2D Viewer")
	widget.resize(800, 800)
	widget.show()
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
