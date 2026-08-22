#!/usr/bin/env python
"""EMScene3D - 3D scene viewer with hierarchical scene graph using PyGfx."""

import numpy as np
import weakref
import math

import pygfx as gfx
from rendercanvas.qt import QRenderWidget

from PySide6 import QtCore, QtGui, QtWidgets
from EMAN3.gui.valslider import ValSlider, StringBox


# ---------------------------------------------------------------------------
# Scene node hierarchy (simplified, no EMAN2 deps)
# ---------------------------------------------------------------------------

from EMAN3.transform import Transform as EMANTransform

class SceneNode:
	"""Base class for scene graph nodes."""
	name_counter = 0

	def __init__(self, name=None, parent=None):
		SceneNode.name_counter += 1
		self._label = name or (self.__class__.__name__ + "_%d" % SceneNode.name_counter)
		self._parent = None
		self._children = []
		self._visible = True
		self._selected = False
		self._gfx_node = None  # The pygfx object in the scene
		self._color = (0.8, 0.8, 0.8, 1.0)
		self._position = (0.0, 0.0, 0.0)
		self._scale = (1.0, 1.0, 1.0)
		self._rotation = (0.0, 0.0, 0.0)  # EMAN convention: az, alt, phi (degrees)
		if parent:
			parent.add_child(self)

	@staticmethod
	def next_name(prefix="Object"):
		SceneNode.name_counter += 1
		return "%s_%d" % (prefix, SceneNode.name_counter)

	# -- Properties --
	@property
	def label(self):
		return self._label

	@label.setter
	def label(self, value):
		self._label = value

	@property
	def parent(self):
		return self._parent

	@property
	def children(self):
		return list(self._children)

	@property
	def visible(self):
		return self._visible

	@visible.setter
	def visible(self, value):
		self._visible = bool(value)
		if self._gfx_node is not None:
			self._gfx_node.visible = self._visible

	@property
	def selected(self):
		return self._selected

	@selected.setter
	def selected(self, value):
		self._selected = bool(value)

	# -- Hierarchy --
	def add_child(self, node):
		if node is self:
			raise ValueError("Cannot add self as child")
		if node in self._children:
			return
		if node._parent:
			node._parent.remove_child(node)
		node._parent = self
		self._children.append(node)

	def remove_child(self, node):
		if node not in self._children:
			return
		node._parent = None
		self._children.remove(node)

	def remove_all_children(self):
		for child in list(self._children):
			child._parent = None
		self._children.clear()

	def get_all_nodes(self):
		result = [self]
		for child in self._children:
			result.extend(child.get_all_nodes())
		return result

	def find_node_by_label(self, label):
		if self._label == label:
			return self
		for child in self._children:
			found = child.find_node_by_label(label)
			if found:
				return found
		return None

	# -- Transform helpers (applied to gfx_node.world) --
	def set_position(self, x, y, z):
		self._position = (float(x), float(y), float(z))
		if self._gfx_node:
			self._apply_transform()

	def get_rotation(self):
		return self._rotation

	def set_rotation(self, az, alt, phi):
		"""Set rotation using EMAN convention euler angles (degrees)."""
		self._rotation = (float(az), float(alt), float(phi))
		if self._gfx_node:
			self._apply_transform()

	def _apply_transform(self):
		"""Apply stored position and rotation to gfx_node via local.matrix."""
		if not self._gfx_node:
			return
		pos = np.array(self._position)
		az, alt, phi = self._rotation
		# Build rotation matrix from EMAN euler angles using Transform
		t = EMANTransform()
		t.set_rotation({'type': 'eman', 'az': az, 'alt': alt, 'phi': phi})
		# get_matrix returns 3x4 affine; we only want the 3x3 rotation part
		rot_mat = t.get_matrix()[:3, :3]
		# Embed position + rotation + scale into 4x4 transform matrix
		scale = self._scale[0]
		matrix = np.eye(4)
		matrix[:3, :3] = rot_mat * scale
		matrix[0, 3], matrix[1, 3], matrix[2, 3] = pos
		self._gfx_node.local.matrix = matrix

	def get_position(self):
		return self._position

	def set_scale(self, s):
		self._scale = (float(s), float(s), float(s))
		if self._gfx_node:
			self._apply_transform()

	def get_scale(self):
		return self._scale[0]

	def set_color(self, r, g, b, a=1.0):
		self._color = (float(r), float(g), float(b), float(a))


class SceneRoot(SceneNode):
	"""Root node of the scene graph."""
	def __init__(self):
		super().__init__(name="Scene")


class ShapeNode(SceneNode):
	"""Base class for shape nodes with geometry parameters."""
	pass


# ---------------------------------------------------------------------------
# Shape-specific nodes with type parameters
# ---------------------------------------------------------------------------

class SphereNode(ShapeNode):
	"""Sphere node with radius parameter."""
	def __init__(self, *args, **kwargs):
		radius = kwargs.pop('radius', 0.5)
		super().__init__(*args, **kwargs)
		self._radius = radius

	def set_radius(self, r):
		"""Set sphere radius and rebuild geometry."""
		self._radius = float(r)
		self._rebuild_geometry()

	def _rebuild_geometry(self):
		geo = gfx.sphere_geometry(self._radius, width_segments=32, height_segments=24)
		self._replace_mesh(geo)

	def inspector_controls(self, parent):
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		controls = []
		slider = ValSlider(parent, (0.01, 50.0), "Radius", value=self._radius)
		slider.valueChanged.connect(lambda v: self._on_radius_changed(v, tgt))
		controls.append(("Radius", slider))
		return controls

	def _on_radius_changed(self, v, tgt):
		self.set_radius(v)
		if tgt:
			tgt._request_render()


class CubeNode(ShapeNode):
	"""Cube node with independent width/height/depth."""
	def __init__(self, *args, **kwargs):
		width = kwargs.pop('width', 1.0)
		height = kwargs.pop('height', 1.0)
		depth = kwargs.pop('depth', 1.0)
		super().__init__(*args, **kwargs)
		self._width = width
		self._height = height
		self._depth = depth

	def set_dimensions(self, w, h, d):
		"""Set cube dimensions and rebuild geometry."""
		self._width, self._height, self._depth = float(w), float(h), float(d)
		self._rebuild_geometry()

	def _rebuild_geometry(self):
		geo = gfx.box_geometry(self._width, self._height, self._depth)
		self._replace_mesh(geo)

	def inspector_controls(self, parent):
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		controls = []
		sw = ValSlider(parent, (0.01, 50.0), "Width", value=self._width)
		sw.valueChanged.connect(lambda v: self._on_dim_changed(v, self._height, self._depth, tgt))
		controls.append(("Width", sw))
		sht = ValSlider(parent, (0.01, 50.0), "Height", value=self._height)
		sht.valueChanged.connect(lambda v: self._on_dim_changed(self._width, v, self._depth, tgt))
		controls.append(("Height", sht))
		sd = ValSlider(parent, (0.01, 50.0), "Depth", value=self._depth)
		sd.valueChanged.connect(lambda v: self._on_dim_changed(self._width, self._height, v, tgt))
		controls.append(("Depth", sd))
		return controls

	def _on_dim_changed(self, w, h, d, tgt):
		self.set_dimensions(w, h, d)
		if tgt:
			tgt._request_render()


class CylinderNode(ShapeNode):
	"""Cylinder node with radius and height."""
	def __init__(self, *args, **kwargs):
		radius = kwargs.pop('radius', 0.5)
		height = kwargs.pop('height', 1.0)
		super().__init__(*args, **kwargs)
		self._radius = radius
		self._height = height

	def set_parameters(self, r, h):
		"""Set cylinder radius and height."""
		self._radius, self._height = float(r), float(h)
		self._rebuild_geometry()

	def _rebuild_geometry(self):
		geo = gfx.cylinder_geometry(radius_bottom=self._radius, radius_top=self._radius,
			height=self._height, radial_segments=32, height_segments=1)
		self._replace_mesh(geo)

	def inspector_controls(self, parent):
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		controls = []
		sr = ValSlider(parent, (0.01, 50.0), "Radius", value=self._radius)
		sr.valueChanged.connect(lambda v: self._on_param_changed(v, self._height, tgt))
		controls.append(("Radius", sr))
		sht = ValSlider(parent, (0.01, 50.0), "Height", value=self._height)
		sht.valueChanged.connect(lambda v: self._on_param_changed(self._radius, v, tgt))
		controls.append(("Height", sht))
		return controls

	def _on_param_changed(self, r, h, tgt):
		self.set_parameters(r, h)
		if tgt:
			tgt._request_render()


class ConeNode(ShapeNode):
	"""Cone node with radius and height."""
	def __init__(self, *args, **kwargs):
		radius = kwargs.pop('radius', 0.5)
		height = kwargs.pop('height', 1.0)
		super().__init__(*args, **kwargs)
		self._radius = radius
		self._height = height

	def set_parameters(self, r, h):
		"""Set cone radius and height."""
		self._radius, self._height = float(r), float(h)
		self._rebuild_geometry()

	def _rebuild_geometry(self):
		geo = gfx.cone_geometry(radius=self._radius, height=self._height,
			radial_segments=32, open_ended=False)
		self._replace_mesh(geo)

	def inspector_controls(self, parent):
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		controls = []
		sr = ValSlider(parent, (0.01, 50.0), "Radius", value=self._radius)
		sr.valueChanged.connect(lambda v: self._on_param_changed(v, self._height, tgt))
		controls.append(("Radius", sr))
		sht = ValSlider(parent, (0.01, 50.0), "Height", value=self._height)
		sht.valueChanged.connect(lambda v: self._on_param_changed(self._radius, v, tgt))
		controls.append(("Height", sht))
		return controls

	def _on_param_changed(self, r, h, tgt):
		self.set_parameters(r, h)
		if tgt:
			tgt._request_render()


class LineNode(ShapeNode):
	"""Line node with start and end points."""
	def __init__(self, *args, **kwargs):
		start = kwargs.pop('start', (0.0, 0.0, 0.0))
		end = kwargs.pop('end', (1.0, 1.0, 1.0))
		super().__init__(*args, **kwargs)
		self._start = np.array(start, dtype=np.float32)
		self._end = np.array(end, dtype=np.float32)

	def set_endpoints(self, start, end):
		"""Set line endpoints."""
		self._start = np.array(start, dtype=np.float32)
		self._end = np.array(end, dtype=np.float32)
		self._rebuild_geometry()

	def _rebuild_geometry(self):
		points = np.vstack([self._start, self._end])
		if hasattr(self, '_line') and self._line:
			self._line.geometry.positions.data = points

	def inspector_controls(self, parent):
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		controls = []
		for i, ax in enumerate(['X', 'Y', 'Z']):
			sl = ValSlider(parent, (-50.0, 50.0), "Start"+ax, value=float(self._start[i]))
			sl.valueChanged.connect(lambda v, idx=i: self._on_start_changed(idx, v, tgt))
			controls.append(("Start"+ax, sl))
		for i, ax in enumerate(['X', 'Y', 'Z']):
			e = ValSlider(parent, (-50.0, 50.0), "End"+ax, value=float(self._end[i]))
			e.valueChanged.connect(lambda v, idx=i: self._on_end_changed(idx, v, tgt))
			controls.append(("End"+ax, e))
		return controls

	def _on_start_changed(self, idx, value, tgt):
		self._start[idx] = float(value)
		self._rebuild_geometry()
		if tgt:
			tgt._request_render()

	def _on_end_changed(self, idx, value, tgt):
		self._end[idx] = float(value)
		self._rebuild_geometry()
		if tgt:
			tgt._request_render()


class ArrowNode(ShapeNode):
	"""Arrow node with shaft/head parameters."""
	def __init__(self, *args, **kwargs):
		shaft_radius = kwargs.pop('shaft_radius', 0.08)
		shaft_length = kwargs.pop('shaft_length', 0.7)
		head_radius = kwargs.pop('head_radius', 0.2)
		head_height = kwargs.pop('head_height', 0.3)
		super().__init__(*args, **kwargs)
		self._shaft_radius = shaft_radius
		self._shaft_length = shaft_length
		self._head_radius = head_radius
		self._head_height = head_height

	def set_parameters(self, sr, sl, hr, hh):
		"""Set arrow parameters."""
		self._shaft_radius = float(sr)
		self._shaft_length = float(sl)
		self._head_radius = float(hr)
		self._head_height = float(hh)
		self._rebuild_geometry()

	def _rebuild_geometry(self):
		head_geo = gfx.cone_geometry(radius=self._head_radius, height=self._head_height,
			radial_segments=16, open_ended=True)
		body_geo = gfx.cylinder_geometry(radius_bottom=self._shaft_radius,
			radius_top=self._shaft_radius, height=self._shaft_length,
			radial_segments=12, height_segments=1)
		vs = np.vstack([head_geo.positions.data, body_geo.positions.data])
		is_combined = np.concatenate([head_geo.indices.data,
			body_geo.indices.data + len(head_geo.positions.data)])
		geo = gfx.Geometry(positions=vs, indices=is_combined)
		self._replace_mesh(geo)

	def inspector_controls(self, parent):
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		controls = []
		sr = ValSlider(parent, (0.01, 10.0), "ShaftR", value=self._shaft_radius)
		sr.valueChanged.connect(lambda v: self._on_param_changed(v, self._shaft_length,
			self._head_radius, self._head_height, tgt))
		controls.append(("ShaftRadius", sr))
		sl = ValSlider(parent, (0.01, 50.0), "ShaftL", value=self._shaft_length)
		sl.valueChanged.connect(lambda v: self._on_param_changed(self._shaft_radius, v,
			self._head_radius, self._head_height, tgt))
		controls.append(("ShaftLength", sl))
		hr = ValSlider(parent, (0.01, 10.0), "HeadR", value=self._head_radius)
		hr.valueChanged.connect(lambda v: self._on_param_changed(self._shaft_radius,
			self._shaft_length, v, self._head_height, tgt))
		controls.append(("HeadRadius", hr))
		hh = ValSlider(parent, (0.01, 10.0), "HeadH", value=self._head_height)
		hh.valueChanged.connect(lambda v: self._on_param_changed(self._shaft_radius,
			self._shaft_length, self._head_radius, v, tgt))
		controls.append(("HeadHeight", hh))
		return controls

	def _on_param_changed(self, sr, sl, hr, hh, tgt):
		self.set_parameters(sr, sl, hr, hh)
		if tgt:
			tgt._request_render()


class TextNode(ShapeNode):
	"""Text node with string and font size."""
	def __init__(self, *args, **kwargs):
		text_content = kwargs.pop('text', 'Hello')
		font_size = kwargs.pop('font_size', 20)
		super().__init__(*args, **kwargs)
		self._text_content = text_content
		self._font_size = font_size

	def set_text(self, text):
		"""Set text content."""
		self._text_content = text
		if hasattr(self, '_text_obj') and self._text_obj:
			self._text_obj.set_text(text)

	def set_font_size(self, fs):
		"""Set font size."""
		self._font_size = int(max(4, min(fs, 200)))
		if hasattr(self, '_text_obj') and self._text_obj:
			self._text_obj.font_size = self._font_size

	def inspector_controls(self, parent):
		controls = []
		sb = StringBox(parent, "Text", value=self._text_content)
		sb.valueChanged.connect(lambda v, p=parent: self._on_text_changed(v, p))
		controls.append(("Text", sb))
		fs = ValSlider(parent, (4.0, 200.0), "FontSize", value=float(self._font_size))
		fs.setIntonly(True)
		fs.valueChanged.connect(lambda v, p=parent: self._on_font_changed(int(v), p))
		controls.append(("FontSize", fs))
		return controls

	def _on_text_changed(self, v, parent):
		self.set_text(v)
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		if tgt:
			tgt._request_render()

	def _on_font_changed(self, v, parent):
		self.set_font_size(int(v))
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		if tgt:
			tgt._request_render()


class DataNode(SceneNode):
	"""Data node holding 3D volume array with optional bounding box."""
	def __init__(self, *args, data=None, header=None, **kwargs):
		self._data = np.asarray(data) if data is not None else None
		self._header = dict(header) if header else {}
		self._show_bbox = True
		self._filename = kwargs.pop('filename', '')
		self._volume_index = 0
		self._full_stack = None
		super().__init__(*args, **kwargs)

	@staticmethod
	def next_name(prefix="Data"):
		SceneNode.name_counter += 1
		return "%s_%d" % (prefix, SceneNode.name_counter)

	@property
	def data(self):
		return self._data

	@property
	def header(self):
		return self._header

	def _rebuild_bbox(self):
		if self._data is None or not self._gfx_node:
			return
		d, h, w = self._data.shape
		for child in list(self._gfx_node.children):
			if getattr(child, '_is_bbox', False):
				self._gfx_node.remove(child)
		if not self._show_bbox:
			return
		# Build wireframe from 12 edges only (no face diagonals)
		hw, hh, hd = float(w)/2, float(h)/2, float(d)/2
		verts = [
			(-hw,-hh,-hd), ( hw,-hh,-hd), ( hw, hh,-hd), (-hw, hh,-hd),
			(-hw,-hh, hd), ( hw,-hh, hd), ( hw, hh, hd), (-hw, hh, hd),
		]
		edges = [
			(0,1),(1,2),(2,3),(3,0),  # back face
			(4,5),(5,6),(6,7),(7,4),  # front face
			(0,4),(1,5),(2,6),(3,7),  # connecting edges
		]
		pos = np.array([verts[i] for e in edges for i in e], dtype=np.float32)
		geo = gfx.Geometry(positions=pos)
		mat = gfx.LineMaterial(color=(0.3, 0.6, 1.0))
		bbox = gfx.Lines(geo, mat)
		bbox._is_bbox = True
		self._gfx_node.add(bbox)

	def set_show_bbox(self, val):
		self._show_bbox = bool(val)
		self._rebuild_bbox()
		if self._gfx_node:
			for child in self._gfx_node.children:
				if getattr(child, '_is_bbox', False):
					child.visible = self._show_bbox

	def inspector_controls(self, parent):
		controls = []
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		info = QtWidgets.QLabel(parent)
		if self._data is not None:
			if self._full_stack is not None:
				info.setText("Shape: %s | dtype: %s | nimg: %d" % (
					str(self._data.shape), self._data.dtype.name, len(self._full_stack)))
			else:
				info.setText("Shape: %s | dtype: %s" % (str(self._data.shape),
					self._data.dtype.name))
		else:
			info.setText("No data loaded.")
		info.setWordWrap(True)
		info.setStyleSheet("QLabel { color: gray; font-size: 9px; }")
		controls.append(("Info", info))
		# Filename display and file browser button
		filename_widget = QtWidgets.QWidget()
		filename_layout = QtWidgets.QHBoxLayout(filename_widget)
		filename_layout.setContentsMargins(0, 0, 0, 0)
		self._filename_box = StringBox(filename_widget, "File", value=self._filename)
		filename_layout.addWidget(self._filename_box)
		file_btn = QtWidgets.QPushButton("Browse...")
		file_btn.clicked.connect(lambda: self._on_browse_file(tgt))
		filename_layout.addWidget(file_btn)
		controls.append(("File", filename_widget))
		# Volume index selector
		if self._full_stack is not None:
			idx_widget = QtWidgets.QWidget()
			idx_layout = QtWidgets.QHBoxLayout(idx_widget)
			idx_layout.setContentsMargins(0, 0, 0, 0)
			self._index_spin = QtWidgets.QSpinBox()
			self._index_spin.setRange(0, len(self._full_stack) - 1)
			self._index_spin.setValue(self._volume_index)
			label = QtWidgets.QLabel("N:")
			idx_layout.addWidget(label)
			idx_layout.addWidget(self._index_spin)
			self._index_spin.valueChanged.connect(lambda v: self._on_index_changed(v, tgt))
			controls.append(("Volume", idx_widget))
		cb = QtWidgets.QCheckBox("Show Bounding Box")
		cb.setChecked(self._show_bbox)
		cb.toggled.connect(lambda v: self._on_bbox_toggled(v, tgt))
		controls.append(("BBox", cb))
		return controls

	def _on_bbox_toggled(self, val, tgt):
		self.set_show_bbox(val)
		if tgt:
			tgt._request_render()

	def load_from_file(self, filepath):
		"""Load 3D volume data from an image file using ImageIO."""
		try:
			from EMAN3.io.imageio import ImageIO
			io = ImageIO(filepath, 'r')
			if io.nimg > 1:
				data, headers = io.read_images()
				# Store full stack for volume selection
				self._full_stack = data.astype(np.float32)
				self._volume_index = 0
				# Headers is a list of dicts
				if isinstance(headers, (list, tuple)) and len(headers) > 0:
					self._header = dict(headers[0])
				else:
					self._header = {}
			else:
				data, header = io.read_image(0)
				if data.ndim == 3:
					self._data = data.astype(np.float32)
				elif data.ndim == 2:
					self._data = data[np.newaxis, :, :].astype(np.float32)
				else:
					print("Cannot load data from file: unsupported dimensionality")
					return
				if isinstance(header, dict):
					self._header = dict(header)
			io.close()
			self._filename = filepath
			# Extract current volume index
			self._extract_volume(self._volume_index)
			self._rebuild_bbox()
		except Exception as e:
			import traceback
			traceback.print_exc()
			print("Error loading file %s: %s" % (filepath, e))

	def _extract_volume(self, idx):
		"""Extract volume at given index from stack, clamping to valid range."""
		if self._full_stack is not None:
			n = len(self._full_stack)
			idx = max(0, min(idx, n - 1))
			self._volume_index = idx
			self._data = self._full_stack[idx].astype(np.float32)
		else:
			# Single volume case
			self._volume_index = 0

	def _on_index_changed(self, idx, tgt):
		"""Handle volume index change from spin box."""
		self._extract_volume(idx)
		self._rebuild_bbox()
		for child in self._children:
			if isinstance(child, (IsoSurfaceNode, VolumeSliceNode, VolumeRenderNode)):
				if isinstance(child, IsoSurfaceNode) or isinstance(child, VolumeRenderNode):
					child._rebuild_volume()
				else:
					child._rebuild_slice()
		if tgt:
			tgt._request_render()

	def _on_browse_file(self, tgt):
		"""Open file dialog to load volume data."""
		filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
			None, "Load Volume Data", "",
			"Image Files (*.hed *.spi *.em *.mrc *.st *.stk);;All Files (*)")
		if filepath:
			self.load_from_file(filepath)
			# Update the filename box
			self._filename_box.setValue(self._filename, quiet=1)
			# Notify any volume children to rebuild
			for child in self._children:
				if isinstance(child, (IsoSurfaceNode, VolumeSliceNode, VolumeRenderNode)):
					if isinstance(child, IsoSurfaceNode) or isinstance(child, VolumeRenderNode):
						child._rebuild_volume()
					else:
						child._rebuild_slice()
			if tgt:
				tgt._request_render()


class IsoSurfaceNode(ShapeNode):
	"""Isosurface rendered from parent DataNode using VolumeIsoMaterial."""
	def __init__(self, *args, threshold=0.5, color=(1.0, 0.4, 0.2), opacity=1.0,
		**kwargs):
		self._threshold = float(threshold)
		self._color = color
		self._opacity = float(opacity)
		super().__init__(*args, **kwargs)

	def _get_data(self):
		node = self._parent
		while node is not None:
			if isinstance(node, DataNode) and node._data is not None:
				return node._data
			node = node._parent
		return None

	def _make_color_map(self):
		"""Create a 1D color map texture from stored RGB color."""
		c = gfx.Color((self._color[0], self._color[1], self._color[2]))
		color_data = np.array([[c.r, c.g, c.b, c.a]], dtype=np.float32)
		return gfx.Texture(color_data, dim=1)

	def _rebuild_volume(self):
		data = self._get_data()
		if data is None or not self._gfx_node:
			return
		for child in list(self._gfx_node.children):
			if isinstance(child, gfx.Volume):
				self._gfx_node.remove(child)
		clim = (float(data.min()), float(data.max()))
		mat = gfx.VolumeIsoMaterial(
			threshold=self._threshold,
			clim=clim,
			map=self._make_color_map()
		)
		geo = gfx.Geometry(grid=gfx.Texture(data[..., np.newaxis], dim=3))
		vol = gfx.Volume(geo, mat)
		# Offset volume so its center aligns with the group origin.
		# pygfx 3D texture maps as (z, y, x) in world space.
		# data.shape is (d, h, w); texture adds channel dim -> (d, h, w, 1)
		d, h, w = data.shape
		vol.local.position = (-w / 2.0, -h / 2.0, -d / 2.0)
		self._gfx_node.add(vol)

	def set_threshold(self, v):
		self._threshold = float(v)
		for child in self._gfx_node.children if self._gfx_node else []:
			if isinstance(child, gfx.Volume) and hasattr(child.material,
				'threshold'):
				child.material.threshold = self._threshold
				break

	def inspector_controls(self, parent):
		controls = []
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		data = self._get_data()
		hr_min = float(data.min()) if data is not None else 0.0
		hr_max = float(data.max()) if data is not None else 1.0
		# Histogram with draggable threshold bar
		self._hist_widget = IsoHistogram(parent)
		if data is not None:
			bins = np.histogram(data, bins=256, range=(hr_min, hr_max))
			self._hist_widget.set_histogram(bins[0], hr_min, hr_max, self._threshold)
		controls.append(("Histogram", self._hist_widget))
		# Threshold slider
		self._thr_slider = ValSlider(parent, (hr_min, hr_max), "Thr:", value=self._threshold)
		self._hist_widget.thresholdChanged.connect(
			lambda v: self._on_threshold_changed(v, tgt))
		self._thr_slider.valueChanged.connect(lambda v: self._on_threshold_changed(v, tgt))
		controls.append(("Threshold", self._thr_slider))
		well = QtWidgets.QPushButton()
		well.setFixedSize(40, 30)
		r, g, b = int(self._color[0]*255), int(self._color[1]*255), int(self._color[2]*255)
		well.setStyleSheet("background-color: rgb(%d,%d,%d); border: 1px solid #555;" % (r, g, b))
		well.clicked.connect(lambda: self._on_color_pick(parent))
		controls.append(("Color", well))
		return controls

	def _on_threshold_changed(self, v, tgt):
		self.set_threshold(v)
		# Sync histogram bar position
		if hasattr(self, '_hist_widget') and self._hist_widget:
			self._hist_widget._threshold = v
			self._hist_widget.update()
		# Sync slider value
		if hasattr(self, '_thr_slider') and self._thr_slider:
			self._thr_slider.setValue(v, quiet=1)
		if tgt:
			tgt._request_render()

	def _on_color_pick(self, parent):
		qcolor = QtGui.QColor(*[int(c * 255) for c in self._color[:3]])
		new_color = QtWidgets.QColorDialog.getColor(qcolor, parent)
		if new_color.isValid():
			self._color = (
				new_color.red() / 255.0,
				new_color.green() / 255.0,
				new_color.blue() / 255.0)
			self._rebuild_volume()
			tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
			if tgt:
				tgt._request_render()


class ScatterPlotNode(ShapeNode):
	"""3D scatter plot rendered as spheres from N×4 data (x,y,z,amplitude)."""
	def __init__(self, *args, point_size=0.1, color_map="viridis", **kwargs):
		self._data = None  # N×4 array
		self._point_size = float(point_size)
		self._color_map_name = color_map
		super().__init__(*args, **kwargs)

	def set_data(self, data):
		"""Set scatter plot data. data should be Nx4 (x,y,z,amplitude)."""
		if data is not None:
			data = np.asarray(data, dtype=np.float32)
			if data.ndim != 2 or data.shape[1] < 3:
				raise ValueError("Scatter plot data must be Nx3 or Nx4")
			if data.shape[1] == 3:
				# Add unit amplitude column
				data = np.column_stack([data, np.ones(len(data), dtype=np.float32)])
			self._data = data
		else:
			self._data = None
		self._rebuild()

	def _amplitude_to_colors(self):
		"""Map amplitude column to RGBA colors using a colormap."""
		if self._data is None or len(self._data) == 0:
			return np.zeros((0, 4), dtype=np.float32)
		a = self._data[:, 3]
		amin, amax = float(a.min()), float(a.max())
		if amax - amin < 1e-9:
			norm = np.zeros_like(a)
		else:
			norm = (a - amin) / (amax - amin)
		# Try matplotlib colormap, fall back to viridis-like gradient
		try:
			from matplotlib import cm as mpl_cm
			cmap_func = mpl_cm.get_cmap(self._color_map_name if self._color_map_name != "viridis" else "viridis")
			colors = cmap_func(norm).astype(np.float32)
		except Exception:
			# Simple fallback: blue→cyan→yellow gradient
			c1, c2, c3 = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0]), np.array([1.0, 1.0, 0.0])
			colors_rgb = np.zeros((len(norm), 3), dtype=np.float32)
			for i, v in enumerate(norm):
				if v < 0.5:
					t = v * 2.0
					colors_rgb[i] = c1 + t * (c2 - c1)
			else:
				t = (v - 0.5) * 2.0
				colors_rgb[i] = c2 + t * (c3 - c2)
			colors = np.column_stack([colors_rgb, np.ones(len(norm), dtype=np.float32)])
		if colors.shape[1] < 4:
			colors = np.column_stack([colors, np.ones(len(colors), dtype=np.float32)])
		return colors.astype(np.float32)

	def _rebuild(self):
		"""Rebuild the Points object."""
		if not self._gfx_node:
			return
		for child in list(self._gfx_node.children):
			self._gfx_node.remove(child)
		if self._data is None or len(self._data) == 0:
			return
		pos = self._data[:, :3]
		colors = self._amplitude_to_colors()
		geo = gfx.Geometry(
			positions=pos,
			colors=colors
		)
		mat = gfx.PointsMaterial(size=self._point_size)
		points = gfx.Points(geo, mat)
		self._gfx_node.add(points)

	def set_point_size(self, v):
		self._point_size = float(v)
		for child in self._gfx_node.children if self._gfx_node else []:
			if isinstance(child, gfx.Points):
				child.material.size = self._point_size
				break

	def inspector_controls(self, parent):
		controls = []
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		ps = ValSlider(parent, (0.01, 2.0), "Point Size:", value=self._point_size)
		ps.valueChanged.connect(lambda v: self._on_size_changed(v, tgt))
		controls.append(("Point Size", ps))
		# File info display
		info = QtWidgets.QLabel("Points: %d" % len(self._data) if self._data is not None else "No data")
		controls.append(("Info", info))
		return controls

	def _on_size_changed(self, v, tgt):
		self.set_point_size(v)
		if tgt:
			tgt._request_render()


class VolumeRenderNode(ShapeNode):
	"""Direct volume rendering from parent DataNode using VolumeRayMaterial."""
	def __init__(self, *args, opacity=0.5, clim=None, **kwargs):
		self._opacity = float(opacity)
		self._clim = tuple(clim) if clim is not None else None
		super().__init__(*args, **kwargs)

	def _get_data(self):
		node = self._parent
		while node is not None:
			if isinstance(node, DataNode) and node._data is not None:
				return node._data
			node = node._parent
		return None

	def _make_color_map(self):
		"""Create a grayscale 1D colormap for volume rendering."""
		stops = np.linspace(0, 1, 256)
		ramp = np.column_stack([stops, stops, stops, np.ones_like(stops)])
		return gfx.Texture(ramp.astype(np.float32), dim=1)

	def _rebuild_volume(self):
		data = self._get_data()
		if data is None or not self._gfx_node:
			return
		for child in list(self._gfx_node.children):
			if isinstance(child, gfx.Volume):
				self._gfx_node.remove(child)
		d, h, w = data.shape
		# Initialize clim from data range if not already set
		if self._clim is None:
			self._clim = (float(data.min()), float(data.max()))
		mat = gfx.VolumeRayMaterial(
			opacity=self._opacity,
			clim=self._clim,
			map=self._make_color_map()
		)
		geo = gfx.Geometry(grid=gfx.Texture(data[..., np.newaxis], dim=3))
		vol = gfx.Volume(geo, mat)
		vol.local.position = (-w / 2.0, -h / 2.0, -d / 2.0)
		self._gfx_node.add(vol)

	def set_opacity(self, v):
		self._opacity = float(v)
		for child in self._gfx_node.children if self._gfx_node else []:
			if isinstance(child, gfx.Volume) and hasattr(child.material, 'opacity'):
				child.material.opacity = self._opacity
				break

	def set_clim(self, low, high):
		self._clim = (float(low), float(high))
		for child in self._gfx_node.children if self._gfx_node else []:
			if isinstance(child, gfx.Volume) and hasattr(child.material, 'clim'):
				child.material.clim = self._clim
				break

	def inspector_controls(self, parent):
		controls = []
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		data = self._get_data()
		if data is not None:
			hr_min, hr_max = float(data.min()), float(data.max())
		else:
			hr_min, hr_max = 0.0, 1.0
		op_sl = ValSlider(parent, (0.0, 1.0), "Opacity:", value=self._opacity)
		op_sl.valueChanged.connect(lambda v: self._on_prop_changed('opacity', v, tgt))
		controls.append(("Opacity", op_sl))
		cl_low = self._clim[0] if self._clim is not None else hr_min
		cl_high = self._clim[1] if self._clim is not None else hr_max
		lo_sl = ValSlider(parent, (hr_min, hr_max), "ClimMin:", value=cl_low)
		lo_sl.valueChanged.connect(lambda v: self._on_clim_changed('min', v, tgt))
		controls.append(("Clim Min", lo_sl))
		hi_sl = ValSlider(parent, (hr_min, hr_max), "ClimMax:", value=cl_high)
		hi_sl.valueChanged.connect(lambda v: self._on_clim_changed('max', v, tgt))
		controls.append(("Clim Max", hi_sl))
		return controls

	def _on_prop_changed(self, prop, v, tgt):
		if prop == 'opacity':
			self.set_opacity(v)
		if tgt:
			tgt._request_render()

	def _on_clim_changed(self, which, v, tgt):
		if which == 'min':
			new_low = float(v)
			new_high = self._clim[1] if self._clim is not None else 1.0
		else:
			new_low = self._clim[0] if self._clim is not None else 0.0
			new_high = float(v)
		self.set_clim(new_low, new_high)
		if tgt:
			tgt._request_render()


class VolumeSliceNode(ShapeNode):
	"""Volume slice rendered from parent DataNode using VolumeSliceMaterial."""
	def __init__(self, *args, offset=0.0, color=(1.0, 1.0, 1.0), **kwargs):
		self._offset = float(offset)
		self._color = color
		super().__init__(*args, **kwargs)

	def _get_data(self):
		node = self._parent
		while node is not None:
			if isinstance(node, DataNode) and node._data is not None:
				return node._data
			node = node._parent
		return None

	def _make_color_map(self):
		c = gfx.Color((self._color[0], self._color[1], self._color[2]))
		color_data = np.array([[c.r, c.g, c.b, c.a]], dtype=np.float32)
		return gfx.Texture(color_data, dim=1)

	def _compute_plane_normal(self):
		"""Compute the slice plane normal from Euler angles in data/grid space."""
		az, alt, phi = self._rotation
		t = EMANTransform()
		t.set_rotation({'type': 'eman', 'az': az, 'alt': alt, 'phi': phi})
		rot_mat = t.get_matrix()[:3, :3]
		normal = rot_mat @ np.array([0, 0, 1], dtype=np.float32)
		norm = float(np.linalg.norm(normal))
		if norm < 1e-6:
			normal = np.array([0, 0, 1], dtype=np.float32)
		else:
			normal /= norm
		return normal

	def _rebuild_slice(self):
		data = self._get_data()
		if data is None or not self._gfx_node:
			return
		for child in list(self._gfx_node.children):
			if isinstance(child, gfx.Volume):
				self._gfx_node.remove(child)
		d, h, w = data.shape
		clim = (float(data.min()), float(data.max()))
		normal = self._compute_plane_normal()
		a, b, c = normal[0], normal[1], normal[2]
		dd = -self._offset
		mat = gfx.VolumeSliceMaterial(
			clim=clim,
			plane=(a, b, c, dd),
			map=gfx.Texture(np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32), dim=1)
		)
		geo = gfx.Geometry(grid=gfx.Texture(data[..., np.newaxis], dim=3))
		vol = gfx.Volume(geo, mat)
		vol.local.position = (-w / 2.0, -h / 2.0, -d / 2.0)
		self._gfx_node.add(vol)

	def _apply_transform(self):
		"""For slices, only apply position+scale to gfx node; rotation drives plane."""
		if not self._gfx_node:
			return
		pos = np.array(self._position)
		scale = self._scale[0]
		matrix = np.eye(4) * scale
		matrix[3, 3] = 1.0
		matrix[0, 3], matrix[1, 3], matrix[2, 3] = pos
		self._gfx_node.local.matrix = matrix
		self._rebuild_slice()

	def set_offset(self, v):
		"""Set slice offset along the plane normal."""
		self._offset = float(v)
		for child in self._gfx_node.children if self._gfx_node else []:
			if isinstance(child, gfx.Volume) and hasattr(child.material,
				'plane'):
				normal = self._compute_plane_normal()
				a, b, cc = normal[0], normal[1], normal[2]
				child.material.plane = (a, b, cc, -self._offset)

	def inspector_controls(self, parent):
		controls = []
		tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
		data = self._get_data()
		if data is not None:
			hr_min, hr_max = float(data.min()), float(data.max())
		else:
			hr_min, hr_max = 0.0, 1.0
		sr = ValSlider(parent, (hr_min, hr_max), "Offset", value=self._offset)
		sr.valueChanged.connect(lambda v: self._on_offset_changed(v, tgt))
		controls.append(("Offset", sr))
		well = QtWidgets.QPushButton()
		well.setFixedSize(40, 30)
		r, g, b = int(self._color[0]*255), int(self._color[1]*255), int(self._color[2]*255)
		well.setStyleSheet("background-color: rgb(%d,%d,%d); border: 1px solid #555;" % (r, g, b))
		well.clicked.connect(lambda: self._on_color_pick(parent))
		controls.append(("Color", well))
		return controls

	def _on_offset_changed(self, v, tgt):
		self.set_offset(v)
		if tgt:
			tgt._request_render()

	def _on_color_pick(self, parent):
		qcolor = QtGui.QColor(*[int(c * 255) for c in self._color[:3]])
		new_color = QtWidgets.QColorDialog.getColor(qcolor, parent)
		if new_color.isValid():
			self._color = (
				new_color.red() / 255.0,
				new_color.green() / 255.0,
				new_color.blue() / 255.0)
			self._rebuild_slice()
			tgt = parent._get_tgt() if hasattr(parent, '_get_tgt') else None
			if tgt:
				tgt._request_render()


class IsoHistogram(QtWidgets.QWidget):
	"""Simple histogram widget with draggable threshold bar."""
	thresholdChanged = QtCore.Signal(float)

	def __init__(self, parent):
		super().__init__(parent)
		self.setMinimumSize(260, 130)
		self._hist_data = None
		self._min_val = 0.0
		self._max_val = 1.0
		self._threshold = 0.5
		self._dragging = False
		self.setMouseTracking(True)

	def set_histogram(self, data, min_val, max_val, threshold):
		"""Set histogram bin counts and value range."""
		self._hist_data = np.array(data).astype(np.float32)
		self._min_val = float(min_val)
		self._max_val = float(max_val)
		self._threshold = float(threshold)
		self.update()

	def paintEvent(self, event):
		if self._hist_data is None:
			return
		painter = QtGui.QPainter(self)
		painter.fillRect(0, 0, self.width(), self.height(), QtGui.QColor(20, 20, 20))
		w = self.width()
		h = self.height() - 20
		norm = float(np.max(self._hist_data)) if np.max(self._hist_data) > 0 else 1.0
		# Draw histogram bars
		painter.setPen(QtGui.QColor(180, 180, 180))
		bins = len(self._hist_data)
		bar_w = w / max(bins, 1)
		for i in range(bins):
			bar_h = int(self._hist_data[i] / norm * h)
			painter.drawLine(i * bar_w, h, i * bar_w, h - bar_h)
		# Draw threshold line
		if self._max_val != self._min_val:
			x_pos = int((self._threshold - self._min_val) / (self._max_val - self._min_val) * w)
		else:
			x_pos = w // 2
		x_pos = max(0, min(x_pos, w))
		painter.setPen(QtGui.QColor(255, 80, 80))
		painter.drawLine(x_pos, 0, x_pos, h)
		# Draw threshold value text
		painter.end()

	def mousePressEvent(self, event):
		if event.button() == QtCore.Qt.MouseButton.LeftButton:
			self._dragging = True
			self._update_threshold_from_event(event)
			event.accept()

	def mouseMoveEvent(self, event):
		if self._dragging:
			self._update_threshold_from_event(event)
			event.accept()

	def mouseReleaseEvent(self, event):
		self._dragging = False
		event.accept()

	def _update_threshold_from_event(self, event):
		x = event.position().x()
		w = self.width()
		t_frac = max(0.0, min(1.0, x / w))
		self._threshold = self._min_val + t_frac * (self._max_val - self._min_val)
		self.update()
		self.thresholdChanged.emit(self._threshold)


# ---------------------------------------------------------------------------
# Shape helper methods (shared across types)  
# ---------------------------------------------------------------------------

def _replace_mesh(self, new_geo):
	"""Replace mesh geometry while preserving material."""
	if not self._gfx_node:
		return
	for child in self._gfx_node.children:
		if isinstance(child, gfx.Mesh):
			child.geometry = new_geo
			break

ShapeNode._replace_mesh = _replace_mesh

# ---------------------------------------------------------------------------
# Canvas for event forwarding  
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class EMScene3DWidget(QtWidgets.QWidget):
	"""3D scene viewer with PyGfx rendering and a hierarchical scene graph."""

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setFocusPolicy(QtCore.Qt.StrongFocus)
		self.setMouseTracking(True)

		self._canvas = _EMCanvas(self, parent=self)
		layout = QtWidgets.QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)
		layout.addWidget(self._canvas)

		self._scene_root = SceneRoot()
		self._selected_node = self._scene_root
		self._inspector = None
		self._bg_color = (0.0, 0.0, 0.0, 1.0)
		self._mouse_mode = "rotate"

		self._orbit_controller = None
		self._camera = None
		self._renderer = None
		self._scene = None
		self._bg_obj = None
		self._light_pos = None
		self._ambient = 0.3

		self._last_x = None
		self._last_y = None
		self._shift_held = False

		# Root group created early so shapes can be added before scene init
		self._root_group = gfx.Group()

		self._dirty = True
		self._redraw = False

		self._setup_gfx()

		# Defer scene init to after first paint (renderer needs valid size)
		QtCore.QTimer.singleShot(0, self._init_scene)

	def _setup_gfx(self):
		"""Set up PyGfx renderer and basic scene."""
		self._renderer = gfx.WgpuRenderer(self._canvas)

		self._scene = gfx.Scene()

		self._bg_mat = gfx.BackgroundMaterial((0.5, 0.5, 0.5, 1.0))
		self._bg_obj = gfx.Background(None, self._bg_mat)
		self._scene.add(self._bg_obj)

		self._request_render()

	def _init_scene(self):
		"""One-time scene graph initialization after renderer is ready."""
		try:
			w, h = self._renderer.logical_size[0], self._renderer.logical_size[1]
			if w == 0:
				w, h = 800, 600

			# Camera
			import numpy as np
			self._camera = gfx.PerspectiveCamera(20)
			self._camera.world.position = (0,0,10.0)
			self._camera.look_at(np.array([0, 0, 0]))

			# Lighting for MeshStandardMaterial
			self._ambient = 0.3
			self._ambient_light = gfx.AmbientLight(color=(self._ambient, self._ambient, self._ambient))
			self._scene.add(self._ambient_light)

			# World-space point light (adjustable via inspector)
			self._light_pos = (2.0, 3.0, 2.0)
			self._point_light = gfx.PointLight(color=(1, 1, 1), intensity=0.6)
			self._point_light.local.position = self._light_pos
			self._scene.add(self._point_light)

			# Camera-following directional light (headlight) for consistent viewer-direction illumination
			self._dir_light = gfx.DirectionalLight(color=(1, 1, 1), intensity=0.6)

			self._orbit_controller = gfx.OrbitController()
			self._orbit_controller.target = (0, 0, 0)
			self._orbit_controller.add_camera(self._camera)
			self._orbit_controller.register_events(self._renderer)

			# Add camera to scene so renderer traverses it (required for parented lights)
			self._scene.add(self._camera)

			# Parent directional light AND its target to camera for proper headlight behavior
			self._dir_light.local.position = (0.0, 0.0, 0.0)
			self._camera.add(self._dir_light)
			self._dir_light.target.local.position = (0.0, 0.0, -1.0)
			self._camera.add(self._dir_light.target)

			# Add root group to scene
			self._scene.add(self._root_group)

			self._request_render()
		except Exception as e:
			import traceback
			print("Scene init error:", e)
			traceback.print_exc()

	def _request_render(self):
		self._dirty = True
		if self._canvas is None:
			return
		self._canvas.request_draw(self._render_callback)

	def _render_callback(self):
		if not getattr(self, '_dirty', False):
			return
		self._dirty = False
		try:
			self._update_gfx_nodes()
			self._renderer.render(self._scene, self._camera)
		except Exception as e:
			import traceback
			print("Render error:", e)
			traceback.print_exc()

	def _update_gfx_nodes(self):
		"""Sync visibility of all gfx nodes."""
		for node in self._scene_root.get_all_nodes():
			if node._gfx_node is not None:
				node._gfx_node.visible = node._visible

	# -- Shape creation helpers --

	def add_cube(self, name=None, position=(0, 0, 0), scale=1.0, color=(0.8, 0.4, 0.2, 1.0), parent=None):
		"""Add a cube to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Cube")

		node = CubeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color
		node.set_dimensions(scale, scale, scale)

		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		geo = gfx.box_geometry(node._width, node._height, node._depth)
		mesh = gfx.Mesh(geo, material)
		group = gfx.Group()
		group.add(mesh)
		node._gfx_node = group
		node._rebuild_geometry()
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		self._request_render()
		return node

	def add_sphere(self, name=None, position=(0, 0, 0), scale=1.0, color=(0.4, 0.6, 0.8, 1.0), parent=None):
		"""Add a sphere to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Sphere")

		node = SphereNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color
		node.set_radius(0.5 * scale)

		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		geo = gfx.sphere_geometry(node._radius, width_segments=32, height_segments=24)
		mesh = gfx.Mesh(geo, material)
		group = gfx.Group()
		group.add(mesh)
		node._gfx_node = group
		node._rebuild_geometry()
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		self._request_render()
		return node

	def add_cylinder(self, name=None, position=(0, 0, 0), scale=1.0, color=(0.6, 0.3, 0.7, 1.0), parent=None):
		"""Add a cylinder to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Cylinder")

		node = CylinderNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color
		node.set_parameters(0.5 * scale, 1.0 * scale)

		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		geo = gfx.cylinder_geometry(radius_bottom=node._radius, radius_top=node._radius,
			height=node._height, radial_segments=32, height_segments=1)
		mesh = gfx.Mesh(geo, material)
		group = gfx.Group()
		group.add(mesh)
		node._gfx_node = group
		node._rebuild_geometry()
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		self._request_render()
		return node

	def add_cone(self, name=None, position=(0, 0, 0), scale=1.0, color=(0.9, 0.3, 0.3, 1.0), parent=None):
		"""Add a cone to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Cone")

		node = ConeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color
		node.set_parameters(0.5 * scale, 1.0 * scale)

		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		geo = gfx.cone_geometry(radius=node._radius, height=node._height,
			radial_segments=32, open_ended=False)
		mesh = gfx.Mesh(geo, material)
		group = gfx.Group()
		group.add(mesh)
		node._gfx_node = group
		node._rebuild_geometry()
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		self._request_render()
		return node

	def add_line(self, name=None, start=(0, 0, 0), end=(1, 1, 1), color=(0.2, 1.0, 0.2, 1.0), parent=None):
		"""Add a line segment to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Line")

		node = LineNode(name=label, start=start, end=end, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		points = np.array([start, end], dtype=np.float32)
		geo = gfx.Geometry(positions=points)
		material = gfx.LineMaterial(color=(color[0], color[1], color[2]), thickness=2.0)
		line_obj = gfx.Line(geo, material)
		node._line = line_obj
		group = gfx.Group()
		group.add(line_obj)
		node._gfx_node = group
		node.set_position(*start)

		self._root_group.add(group)
		self._request_render()
		return node

	def add_arrow(self, name=None, position=(0, 0, 0), scale=1.0, color=(1.0, 0.2, 0.2, 1.0), parent=None):
		"""Add an arrow to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Arrow")

		node = ArrowNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		head_geo = gfx.cone_geometry(radius=node._head_radius, height=node._head_height,
			radial_segments=16, open_ended=True)
		body_geo = gfx.cylinder_geometry(radius_bottom=node._shaft_radius,
			radius_top=node._shaft_radius, height=node._shaft_length,
			radial_segments=12, height_segments=1)
		vs = np.vstack([head_geo.positions.data, body_geo.positions.data])
		is_combined = np.concatenate([head_geo.indices.data,
			body_geo.indices.data + len(head_geo.positions.data)])
		geo = gfx.Geometry(positions=vs, indices=is_combined)
		mesh = gfx.Mesh(geo, material)

		group = gfx.Group()
		group.add(mesh)
		node._gfx_node = group
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		self._request_render()
		return node

	def add_text(self, name=None, text="Hello", position=(0, 0, 0), scale=0.1, color=(1.0, 1.0, 1.0, 1.0), parent=None):
		"""Add 3D text to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Text")

		node = TextNode(name=label, text=text, font_size=20, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		text_obj = gfx.Text(text=text, font_size=20, render_order=999)
		text_obj.material.color = (color[0], color[1], color[2])
		node._text_obj = text_obj
		group = gfx.Group()
		group.add(text_obj)
		node._gfx_node = group
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		self._request_render()
		return node

	def add_axes(self, name="Axes", scale=0.5):
		"""Add X/Y/Z axis indicator arrows."""
		parent = self._selected_node
		# Arrows point +Y by default, so rotate to align with each axis
		x_arrow = self.add_arrow(name=name + "_X", position=(scale * 0.5, 0, 0), scale=scale, color=(1.0, 0.2, 0.2), parent=parent)
		import pylinalg as la
		# Rotate +90 deg around Z to point +X
		q = la.quat_from_rotvec((0, 0, -math.pi / 2))
		x_arrow._gfx_node.local.matrix = la.mat_from_quat(q)
		y_arrow = self.add_arrow(name=name + "_Y", position=(0, scale * 0.5, 0), scale=scale, color=(0.2, 1.0, 0.2), parent=parent)
		z_arrow = self.add_arrow(name=name + "_Z", position=(0, 0, scale * 0.5), scale=scale, color=(0.2, 0.2, 1.0), parent=parent)
		# Rotate -90 deg around X to point +Z
		q = la.quat_from_rotvec((-math.pi / 2, 0, 0))
		z_arrow._gfx_node.local.matrix = la.mat_from_quat(q)
		return x_arrow, y_arrow, z_arrow

	def add_data(self, name=None, data=None, header=None, filename='', parent=None):
		"""Add a DataNode holding 3D volume array."""
		if parent is None:
			parent = self._selected_node
		label = name or DataNode.next_name()
		node = DataNode(name=label, data=data, header=header,
			filename=filename, parent=parent)
		node._color = (0.3, 0.6, 1.0, 1.0)
		group = gfx.Group()
		node._gfx_node = group
		if node._data is not None:
			node._rebuild_bbox()
		self._root_group.add(group)
		self._request_render()
		return node

	def add_isosurface(self, name=None, threshold=0.5, color=(1.0, 0.4, 0.2),
		parent=None):
		"""Add an IsoSurfaceNode that renders isosurface from parent DataNode."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("IsoSurface")
		node = IsoSurfaceNode(name=label, threshold=threshold,
			color=color, parent=parent)
		group = gfx.Group()
		node._gfx_node = group
		node._rebuild_volume()
		self._root_group.add(group)
		self._request_render()
		return node

	def add_slice(self, name=None, offset=0.0, color=(1.0, 1.0, 1.0), parent=None):
		"""Add a VolumeSliceNode that renders a slice through parent DataNode."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Slice")
		node = VolumeSliceNode(name=label, offset=offset,
			color=color, parent=parent)
		group = gfx.Group()
		node._gfx_node = group
		node._rebuild_slice()
		self._root_group.add(group)
		self._request_render()
		return node

	def add_volume_render(self, name=None, opacity=0.5, parent=None):
		"""Add a VolumeRenderNode for direct volume rendering from parent DataNode."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("VolRender")
		node = VolumeRenderNode(name=label, opacity=opacity, parent=parent)
		group = gfx.Group()
		node._gfx_node = group
		node._rebuild_volume()
		self._root_group.add(group)
		self._request_render()
		return node

	def add_scatter_plot(self, name=None, data=None, point_size=0.1, parent=None):
		"""Add a ScatterPlotNode for 3D scatter plot rendering."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Scatter")
		node = ScatterPlotNode(name=label, point_size=point_size, parent=parent)
		group = gfx.Group()
		node._gfx_node = group
		if data is not None:
			node.set_data(data)
		self._root_group.add(group)
		self._request_render()
		return node

	# -- Node management --

	def get_scene_root(self):
		return self._scene_root

	def get_selected_node(self):
		return self._selected_node

	def set_selected_node(self, node):
		if isinstance(node, SceneNode):
			self._selected_node = node

	def remove_node(self, node):
		"""Remove a node and its gfx object from the scene."""
		if node is self._scene_root:
			return False
		if node._gfx_node:
			self._root_group.remove(node._gfx_node)
		if node.parent:
			node.parent.remove_child(node)
		# Also remove children's gfx nodes
		for child in node.get_all_nodes()[1:]:  # Skip self
			if child._gfx_node:
				self._root_group.remove(child._gfx_node)
		self._request_render()
		return True

	def set_bg_color(self, r, g, b):
		self._bg_color = (float(r), float(g), float(b), 1.0)
		if self._bg_obj and self._scene is not None:
			self._bg_obj.material.set_colors(gfx.Color((float(r), float(g), float(b),1.0)))
			self._request_render()
		else: print("set_bg_color with no object present")

	def set_ambient(self, value):
		self._ambient = float(value)
		if self._ambient_light:
			self._ambient_light.color = (self._ambient, self._ambient, self._ambient)
		self._request_render()

	def set_light_position(self, h_angle, v_angle):
		"""Set point light angular position (degrees) and update its world position."""
		dist = 4.0
		h = math.radians(h_angle)
		v = math.radians(v_angle)
		x = dist * math.cos(v) * math.sin(h)
		y = dist * math.sin(v)
		z = dist * math.cos(v) * math.cos(h)
		if y < 0.1:
			y = 0.1
		self._light_pos = (x, y, z)
		if self._point_light:
			self._point_light.local.position = self._light_pos
		self._request_render()

	def set_point_light_intensity(self, value):
		"""Set point light intensity."""
		if self._point_light:
			self._point_light.intensity = float(value)
		self._request_render()

	# -- Inspector --

	def get_inspector(self):
		if self._inspector is None:
			self._inspector = EMScene3DInspector(self)
		return self._inspector

	def show_inspector(self):
		insp = self.get_inspector()
		if not insp.isVisible():
			insp.show()

	def closeEvent(self, event):
		if self._inspector:
			self._inspector.setParent(None)
			self._inspector.deleteLater()
		super().closeEvent(event)

	# -- Mouse handling --

	def _on_mouse_press(self, event):
		self._last_x = event.position().x()
		self._last_y = event.position().y()
		self._shift_held = (event.modifiers() & QtCore.Qt.ShiftModifier) != 0

		if event.button() == QtCore.Qt.MiddleButton:
			self.show_inspector()
			event.accept()
			return

	def _on_mouse_move(self, event):
		self._last_x = event.position().x()
		self._last_y = event.position().y()
		self._sync_camera_to_inspector()
		self._request_render()

	def _on_mouse_release(self, event):
		self._last_x = None
		self._last_y = None
		self._sync_camera_to_inspector()
		self._request_render()

	def _on_wheel(self, event):
		angle = event.angleDelta().y() / 120.0
		# The orbit controller handles zooming by default
		event.accept()
		self._sync_camera_to_inspector()
		self._request_render()

	def _get_camera_distance(self):
		"""Get camera distance from orbit target."""
		import numpy as np
		oc = getattr(self, '_orbit_controller', None)
		if not oc:
			return 3.0
		target = np.array(oc.target)
		pos = np.array(self._camera.world.position[:3])
		return float(np.linalg.norm(pos - target))

	def _set_camera_distance(self, dist):
		"""Set camera distance from orbit target by moving along look direction."""
		import numpy as np
		oc = getattr(self, '_orbit_controller', None)
		if not oc:
			return
		target = np.array(oc.target)
		pos = np.array(self._camera.world.position[:3])
		direction = pos - target
		norm = float(np.linalg.norm(direction))
		if norm < 1e-6:
			norm = 1.0
		direction = direction / norm * max(dist, 0.01)
		self._camera.world.position = tuple(direction + target)
		self._request_render()

	def _sync_camera_to_inspector(self):
		"""Sync inspector camera sliders from current orbit controller state."""
		if self._inspector:
			try:
				self._inspector.dist_slider.setValue(self._get_camera_distance(), quiet=1)
			except AttributeError:
				pass
			cam = getattr(self, '_camera', None)
			if cam:
				try:
					self._inspector.fov_slider.setValue(cam.fov, quiet=1)
				except AttributeError:
					pass

	def _on_key_press(self, event):
		key = event.key()
		if key == QtCore.Qt.Key_R:
			self.set_mouse_mode("rotate")
		elif key == QtCore.Qt.Key_T:
			self.set_mouse_mode("translate")
		elif key == QtCore.Qt.Key_S:
			self.set_mouse_mode("scale")
		elif key == QtCore.Qt.Key_Escape:
			self.set_mouse_mode("select")
		elif key == QtCore.Qt.Key_Delete or key == QtCore.Qt.Key_Backspace:
			self._delete_selected()
		else:
			event.ignore()

	def _delete_selected(self):
		if self._selected_node is not self._scene_root:
			self.remove_node(self._selected_node)
			self._selected_node = self._scene_root
			if self._inspector:
				self._inspector.update_tree()

	def set_mouse_mode(self, mode):
		self._mouse_mode = mode
		# Configure orbit controller based on mode
		if self._orbit_controller:
			if mode == "rotate":
				self._orbit_controller.rot_scale = 1.0
			elif mode == "translate":
				self._orbit_controller.pan_speed = 5.0
			elif mode == "scale":
				pass

	def get_mouse_mode(self):
		return self._mouse_mode


# ---------------------------------------------------------------------------
# Inspector
# ---------------------------------------------------------------------------

class EMScene3DInspector(QtWidgets.QWidget):
	"""Inspector for the 3D scene widget."""

	def __init__(self, widget, parent=None):
		super().__init__(parent)
		self.setWindowTitle("EMScene3D Inspector")
		self.setMinimumSize(400, 500)
		self._widget = weakref.ref(widget)

		self._build_ui()
		self._wire_signals()
		self.update_tree()

	def _get_tgt(self):
		return self._widget() if self._widget else None

	def _build_ui(self):
		main_layout = QtWidgets.QVBoxLayout(self)

		# Tabs for different control panels
		self.tab_widget = QtWidgets.QTabWidget()

		# Tree view tab
		self.tab_widget.addTab(self._build_tree_tab(), "Hierarchy")

		# Tools tab
		# Lights tab (removed redundant Tools tab - Add Object button is in Hierarchy tab)
		self.tab_widget.addTab(self._build_lights_tab(), "Lights")

		# Camera tab
		self.tab_widget.addTab(self._build_camera_tab(), "Camera")

		# Utils tab
		self.tab_widget.addTab(self._build_utils_tab(), "Utils")

		main_layout.addWidget(self.tab_widget)

	def _build_tree_tab(self):
		widget = QtWidgets.QWidget()
		main_layout = QtWidgets.QHBoxLayout(widget)

		# Left side: tree + buttons in a column
		left_widget = QtWidgets.QWidget()
		left_layout = QtWidgets.QVBoxLayout(left_widget)

		# Tree widget showing scene hierarchy
		self.tree = QtWidgets.QTreeWidget()
		self.tree.setHeaderLabel("Scene Hierarchy")
		self.tree.setColumnCount(2)
		self.tree.setHeaderLabels(["Name", "Visible"])
		left_layout.addWidget(self.tree)

		# Buttons row
		btn_layout = QtWidgets.QHBoxLayout()
		self.add_btn = QtWidgets.QPushButton("Add Object")
		self.remove_btn = QtWidgets.QPushButton("Remove Object")
		btn_layout.addWidget(self.add_btn)
		btn_layout.addWidget(self.remove_btn)
		left_layout.addLayout(btn_layout)

		# Mouse mode info (OrbitController handles rotation/panning natively)
		info_label = QtWidgets.QLabel("Left-drag: Rotate | Right-drag: Pan | Scroll: Zoom")
		info_label.setStyleSheet("QLabel { color: gray; font-size: 9px; }")
		left_layout.addWidget(info_label)

		# Right side: properties panel
		right_widget = self._build_properties_panel()

		# Split left/right with a splitter
		splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
		splitter.addWidget(left_widget)
		splitter.addWidget(right_widget)
		splitter.setStretchFactor(0, 1)
		splitter.setStretchFactor(1, 1)

		main_layout.addWidget(splitter)
		return widget

	def _build_properties_panel(self):
		"""Build the properties panel shown when a node is selected."""
		self.prop_widget = QtWidgets.QTabWidget()

		# Common tab: position, rotation, scale, color
		common_page = self._build_common_tab()
		self.prop_widget.addTab(common_page, "Common")

		# Object-specific tab
		object_page = self._build_object_tab()
		self.prop_widget.addTab(object_page, "Object")

		return self.prop_widget

	def _build_common_tab(self):
		"""Build common properties: position, rotation, scale, color."""
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)

		# Position groupbox
		pos_frame = QtWidgets.QGroupBox("Position")
		pos_layout = QtWidgets.QGridLayout(pos_frame)
		self.pos_x_slider = ValSlider(page, (-10.0, 10.0), "X", value=0.0)
		self.pos_y_slider = ValSlider(page, (-10.0, 10.0), "Y", value=0.0)
		self.pos_z_slider = ValSlider(page, (-10.0, 10.0), "Z", value=0.0)
		pos_layout.addWidget(self.pos_x_slider, 0, 0)
		pos_layout.addWidget(self.pos_y_slider, 1, 0)
		pos_layout.addWidget(self.pos_z_slider, 2, 0)
		layout.addWidget(pos_frame)

		# Rotation groupbox (EMAN convention: azimuth, altitude, phi)
		rot_frame = QtWidgets.QGroupBox("Rotation EMAN (deg)")
		rot_layout = QtWidgets.QGridLayout(rot_frame)
		self.rot_az_slider = ValSlider(page, (-180.0, 180.0), "Az", value=0.0)
		self.rot_alt_slider = ValSlider(page, (-90.0, 90.0), "Alt", value=0.0)
		self.rot_phi_slider = ValSlider(page, (-180.0, 180.0), "Phi", value=0.0)
		rot_layout.addWidget(self.rot_az_slider, 0, 0)
		rot_layout.addWidget(self.rot_alt_slider, 1, 0)
		rot_layout.addWidget(self.rot_phi_slider, 2, 0)
		layout.addWidget(rot_frame)

		# Scale groupbox
		scl_frame = QtWidgets.QGroupBox("Scale")
		scl_layout = QtWidgets.QVBoxLayout(scl_frame)
		self.scl_slider = ValSlider(page, (0.1, 20.0), "Scale", value=1.0)
		scl_layout.addWidget(self.scl_slider)
		layout.addWidget(scl_frame)

		# Color well that visually shows the current color
		col_frame = QtWidgets.QGroupBox("Color")
		col_layout = QtWidgets.QHBoxLayout(col_frame)
		self.color_well = QtWidgets.QPushButton()
		self.color_well.setFixedSize(40, 30)
		self.color_well.setStyleSheet("background-color: rgb(128, 128, 128); border: 1px solid #555;")
		col_layout.addWidget(self.color_well)
		layout.addWidget(col_frame)

		layout.addStretch()
		return page

	def _build_object_tab(self):
		"""Build object-specific properties tab."""
		page = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(page)

		# Placeholder label, populated when a shape is selected
		self.object_info_label = QtWidgets.QLabel("Select an object to see properties.")
		self.object_info_label.setWordWrap(True)
		self.object_info_label.setStyleSheet("QLabel { color: gray; }")
		layout.addWidget(self.object_info_label)

		# Object-specific sliders (dynamically wired)
		self.object_controls = QtWidgets.QGroupBox("Parameters")
		self.object_controls_layout = QtWidgets.QVBoxLayout(self.object_controls)
		layout.addWidget(self.object_controls)

		layout.addStretch()
		return page


	def _build_lights_tab(self):
		widget = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(widget)

		# Ambient Light groupbox
		amb_frame = QtWidgets.QGroupBox("Ambient Light")
		amb_layout = QtWidgets.QVBoxLayout(amb_frame)
		self.ambient_slider = ValSlider(widget, (0.0, 1.0), "Intensity", value=0.3)
		amb_layout.addWidget(self.ambient_slider)
		layout.addWidget(amb_frame)

		# Directional Light groupbox - position + intensity together
		dir_frame = QtWidgets.QGroupBox("Directional Light")
		dir_layout = QtWidgets.QVBoxLayout(dir_frame)
		self.dir_h_slider = ValSlider(widget, (0.0, 360.0), "Horizontal", value=45.0)
		self.dir_v_slider = ValSlider(widget, (0.0, 180.0), "Vertical", value=60.0)
		self.dir_i_slider = ValSlider(widget, (0.0, 20.0), "Intensity", value=0.6)
		dir_layout.addWidget(self.dir_h_slider)
		dir_layout.addWidget(self.dir_v_slider)
		dir_layout.addWidget(self.dir_i_slider)
		layout.addWidget(dir_frame)

		# Point Light groupbox - position + intensity together
		point_frame = QtWidgets.QGroupBox("Point Light")
		point_layout = QtWidgets.QVBoxLayout(point_frame)
		self.point_h_slider = ValSlider(widget, (0.0, 360.0), "Horizontal", value=45.0)
		self.point_v_slider = ValSlider(widget, (0.0, 180.0), "Vertical", value=60.0)
		self.point_i_slider = ValSlider(widget, (0.0, 10.0), "Intensity", value=0.6)
		point_layout.addWidget(self.point_h_slider)
		point_layout.addWidget(self.point_v_slider)
		point_layout.addWidget(self.point_i_slider)
		layout.addWidget(point_frame)

		layout.addStretch()
		return widget

	def _build_camera_tab(self):
		widget = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(widget)

		label = QtWidgets.QLabel("Camera")
		label.setFont(QtGui.QFont(label.font().family(), 10, QtGui.QFont.Weight.Bold))
		layout.addWidget(label)

		# FOV slider
		fov_frame = QtWidgets.QGroupBox("Field of View")
		fov_layout = QtWidgets.QVBoxLayout(fov_frame)
		self.fov_slider = ValSlider(widget, (1.0, 60.0), "FOV", value=50.0)
		fov_layout.addWidget(self.fov_slider)
		layout.addWidget(fov_frame)

		# Distance slider (camera distance from origin/target)
		dist_frame = QtWidgets.QGroupBox("Camera Distance")
		dist_layout = QtWidgets.QVBoxLayout(dist_frame)
		self.dist_slider = ValSlider(widget, (0.5, 100.0), "Dist", value=3.0)
		dist_layout.addWidget(self.dist_slider)
		layout.addWidget(dist_frame)

		# Near/Far clip planes
		clip_frame = QtWidgets.QGroupBox("Clip Planes")
		clip_layout = QtWidgets.QVBoxLayout(clip_frame)
		self.near_slider = ValSlider(widget, (0.01, 10.0), "Near", value=0.1)
		self.far_slider = ValSlider(widget, (10.0, 10000.0), "Far", value=500.0)
		clip_layout.addWidget(self.near_slider)
		clip_layout.addWidget(self.far_slider)
		layout.addWidget(clip_frame)

		layout.addStretch()
		return widget

	def _build_utils_tab(self):
		widget = QtWidgets.QWidget()
		layout = QtWidgets.QVBoxLayout(widget)

		label = QtWidgets.QLabel("Utilities")
		label.setFont(QtGui.QFont(label.font().family(), 10, QtGui.QFont.Weight.Bold))
		layout.addWidget(label)

		# Background color
		bg_frame = QtWidgets.QGroupBox("Background")
		bg_layout = QtWidgets.QHBoxLayout(bg_frame)
		self.bg_color_btn = QtWidgets.QPushButton("Set Color")
		self.bg_color_btn.setMinimumWidth(80)
		bg_layout.addWidget(self.bg_color_btn)
		layout.addWidget(bg_frame)

		# Buttons
		self.snapshot_btn = QtWidgets.QPushButton("Save Snapshot")
		layout.addWidget(self.snapshot_btn)

		self.hide_sel_cb = QtWidgets.QCheckBox("Hide Selections")
		layout.addWidget(self.hide_sel_cb)

		layout.addStretch()
		return widget

	def _wire_signals(self):
		tgt = self._get_tgt()
		if not tgt:
			return

		self.add_btn.clicked.connect(self._on_add_dialog)
		self.remove_btn.clicked.connect(self._on_remove_selected)
		self.tree.itemClicked.connect(self._on_tree_click)
		self.tree.itemChanged.connect(self._on_item_changed)

		# Property panel sliders update selected node
		self.pos_x_slider.valueChanged.connect(self._on_pos_x_changed)
		self.pos_y_slider.valueChanged.connect(self._on_pos_y_changed)
		self.pos_z_slider.valueChanged.connect(self._on_pos_z_changed)
		self.rot_az_slider.valueChanged.connect(self._on_rot_az_changed)
		self.rot_alt_slider.valueChanged.connect(self._on_rot_alt_changed)
		self.rot_phi_slider.valueChanged.connect(self._on_rot_phi_changed)
		self.scl_slider.valueChanged.connect(self._on_scale_changed)
		self.color_well.clicked.connect(self._on_color_pick)

		# Ambient light
		self.ambient_slider.valueChanged.connect(self._on_ambient_changed)

		# Directional light - position (angles relative to camera) and intensity
		self.dir_h_slider.valueChanged.connect(self._on_dir_h_changed)
		self.dir_v_slider.valueChanged.connect(self._on_dir_v_changed)
		self.dir_i_slider.valueChanged.connect(self._on_dir_intensity_changed)

		# Point light - position and intensity
		self.point_h_slider.valueChanged.connect(self._on_point_h_changed)
		self.point_v_slider.valueChanged.connect(self._on_point_v_changed)
		self.point_i_slider.valueChanged.connect(self._on_point_intensity_changed)

		# Camera controls
		self.fov_slider.valueChanged.connect(self._on_fov_changed)
		self.dist_slider.valueChanged.connect(self._on_distance_changed)
		self.near_slider.valueChanged.connect(self._on_near_changed)
		self.far_slider.valueChanged.connect(self._on_far_changed)

		self.bg_color_btn.clicked.connect(self._on_bg_color)
		self.snapshot_btn.clicked.connect(self._on_snapshot)

	def _on_add_dialog(self):
		"""Open dialog to add a new shape."""
		dialog = AddNodeDialog(self, self.tree.currentItem())
		if dialog.exec_() == QtWidgets.QDialog.Accepted:
			tgt = self._get_tgt()
			if tgt:
				data = dialog.get_data()
				node = None
				pos = data.get("position", (0, 0, 0))
				scl = data.get("scale", 1.0)
				col = data.get("color", (0.8, 0.4, 0.2, 1.0))
				node_name = data.get("name", None)
				parent_node = self._get_parent_for_insertion()

				shape_type = data["type"]
				if shape_type == "Cube":
					node = tgt.add_cube(name=node_name, position=pos, scale=scl, color=col, parent=parent_node)
				elif shape_type == "Sphere":
					node = tgt.add_sphere(name=node_name, position=pos, scale=scl, color=col, parent=parent_node)
				elif shape_type == "Cylinder":
					node = tgt.add_cylinder(name=node_name, position=pos, scale=scl, color=col, parent=parent_node)
				elif shape_type == "Cone":
					node = tgt.add_cone(name=node_name, position=pos, scale=scl, color=col, parent=parent_node)
				elif shape_type == "Arrow":
					node = tgt.add_arrow(name=node_name, position=pos, scale=scl, color=col, parent=parent_node)
				elif shape_type == "Data":
					node = tgt.add_data(name=node_name, parent=parent_node)
				elif shape_type == "IsoSurface":
					node = tgt.add_isosurface(name=node_name, parent=parent_node)
				elif shape_type == "Slice":
					node = tgt.add_slice(name=node_name, parent=parent_node)
				elif shape_type == "VolumeRender":
					node = tgt.add_volume_render(name=node_name, parent=parent_node)
				elif shape_type == "ScatterPlot":
					node = tgt.add_scatter_plot(name=node_name, data=data.get("scatter_data"), parent=parent_node)

				if node:
					self.update_tree()

	def _get_parent_for_insertion(self):
		"""Get the parent node based on current tree selection."""
		tgt = self._get_tgt()
		if not tgt:
			return None
		current = self.tree.currentItem()
		if current and current.data(0, QtCore.Qt.UserRole) is not None:
			return current.data(0, QtCore.Qt.UserRole)
		return tgt.get_selected_node()

	def _on_remove_selected(self):
		tgt = self._get_tgt()
		if not tgt:
			return
		current = self.tree.currentItem()
		if current:
			node = current.data(0, QtCore.Qt.UserRole)
			if node and node is not tgt.get_scene_root():
				tgt.remove_node(node)
				tgt.set_selected_node(tgt.get_scene_root())
				self.update_tree()

	def _on_item_changed(self, item, column):
		tgt = self._get_tgt()
		if not tgt:
			return
		node = item.data(0, QtCore.Qt.UserRole)
		if node:
			new_vis = (item.checkState(column) == QtCore.Qt.CheckState.Checked)
			node.visible = new_vis
			tgt._request_render()

	def _on_tree_click(self, item, column):
		node = item.data(0, QtCore.Qt.UserRole)
		if node:
			tgt = self._get_tgt()
			if tgt:
				tgt.set_selected_node(node)
				node.selected = True
				for child in tgt.get_scene_root().get_all_nodes():
					if child is not node:
						child.selected = False
				# Update properties panel
				self._update_properties()

	def _update_properties(self):
		"""Update properties panel to reflect currently selected node."""
		tgt = self._get_tgt()
		if not tgt:
			return
		node = tgt.get_selected_node()
		if not node:
			return

		# Update common properties from node state
		pos = node.get_position()
		self.pos_x_slider.setValue(float(pos[0]), quiet=1)
		self.pos_y_slider.setValue(float(pos[1]), quiet=1)
		self.pos_z_slider.setValue(float(pos[2]), quiet=1)

		# Update rotation sliders from stored EMAN euler angles
		rot = node.get_rotation()
		self.rot_az_slider.setValue(float(rot[0]), quiet=1)
		self.rot_alt_slider.setValue(float(rot[1]), quiet=1)
		self.rot_phi_slider.setValue(float(rot[2]), quiet=1)

		scl = node.get_scale()
		self.scl_slider.setValue(float(scl), quiet=1)

		# Update object-specific tab
		if isinstance(node, (ShapeNode, DataNode)):
			info_lines = ["Type: %s" % node.__class__.__name__, "Label: %s" % node.label]
			self.object_info_label.setText("\n".join(info_lines))
			self.object_info_label.setStyleSheet("")

			# Clear existing shape-specific controls
			while self.object_controls_layout.count():
				item = self.object_controls_layout.takeAt(0)
				if item.widget():
					item.widget().deleteLater()

			# Build new controls from the node's inspector_controls method
			if hasattr(node, 'inspector_controls'):
				for label_text, ctrl in node.inspector_controls(self):
					self.object_controls_layout.addWidget(ctrl)
		else:
			self.object_info_label.setText("Scene root - no specific properties.")
			self.object_info_label.setStyleSheet("QLabel { color: gray; }")

		# Update color well visual
		self._update_color_well(node)

	def _on_pos_x_changed(self, value):
		tgt = self._get_tgt()
		if not tgt:
			return
		node = tgt.get_selected_node()
		if node:
			pos = list(node.get_position())
			pos[0] = float(value)
			node.set_position(*pos)
			tgt._request_render()

	def _on_pos_y_changed(self, value):
		tgt = self._get_tgt()
		if not tgt:
			return
		node = tgt.get_selected_node()
		if node:
			pos = list(node.get_position())
			pos[1] = float(value)
			node.set_position(*pos)
			tgt._request_render()

	def _on_pos_z_changed(self, value):
		tgt = self._get_tgt()
		if not tgt:
			return
		node = tgt.get_selected_node()
		if node:
			pos = list(node.get_position())
			pos[2] = float(value)
			node.set_position(*pos)
			tgt._request_render()

	def _on_rot_az_changed(self, value):
		self._apply_rotation()

	def _on_rot_alt_changed(self, value):
		self._apply_rotation()

	def _on_rot_phi_changed(self, value):
		self._apply_rotation()

	def _apply_rotation(self):
		"""Apply EMAN convention euler angles from sliders to selected node."""
		tgt = self._get_tgt()
		if not tgt:
			return
		node = tgt.get_selected_node()
		if not node:
			return
		az = self.rot_az_slider.getValue()
		alt = self.rot_alt_slider.getValue()
		phi = self.rot_phi_slider.getValue()
		node.set_rotation(az, alt, phi)
		tgt._request_render()

	def _on_scale_changed(self, value):
		tgt = self._get_tgt()
		if not tgt:
			return
		node = tgt.get_selected_node()
		if node:
			node.set_scale(float(value))
			tgt._request_render()

	def _on_color_pick(self):
		"""Open color dialog and set selected node color."""
		tgt = self._get_tgt()
		if not tgt:
			return
		node = tgt.get_selected_node()
		if not node:
			return
		current_color = node._color
		qcolor = QtGui.QColor(*[int(c * 255) for c in current_color[:3]])
		new_color = QtWidgets.QColorDialog.getColor(qcolor, self)
		if new_color.isValid():
			r = new_color.red() / 255.0
			g = new_color.green() / 255.0
			b = new_color.blue() / 255.0
			node.set_color(r, g, b)
			if hasattr(node, '_gfx_node') and node._gfx_node:
				for child in node._gfx_node.children:
					if hasattr(child, 'material') and hasattr(child.material, 'color'):
						child.material.color = (r, g, b)
			self._update_color_well(node)
			tgt._request_render()

	def _update_color_well(self, node):
		"""Update the color well visual to match node color."""
		color = node._color
		r, g, b = int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
		self.color_well.setStyleSheet(
			"background-color: rgb(%d, %d, %d); border: 1px solid #555;" % (r, g, b))

	def _on_dir_h_changed(self, value):
		"""Directional light horizontal angle changed."""
		tgt = self._get_tgt()
		if tgt and hasattr(tgt, '_dir_light'):
			v_val = self.dir_v_slider.getValue()
			self._update_dir_light_direction(tgt, value, v_val)

	def _on_dir_v_changed(self, value):
		"""Directional light vertical angle changed."""
		tgt = self._get_tgt()
		if tgt and hasattr(tgt, '_dir_light'):
			h_val = self.dir_h_slider.getValue()
			self._update_dir_light_direction(tgt, h_val, value)

	def _update_dir_light_direction(self, tgt, h_angle, v_angle):
		"""Update directional light target based on angular offsets."""
		h = math.radians(h_angle - 180.0)  # offset so 0=straight ahead
		v = math.radians(v_angle - 90.0)
		x = math.sin(h) * math.cos(v)
		y = math.sin(v)
		z = -math.cos(h) * math.cos(v)
		tgt._dir_light.target.local.position = (x, y, z)
		tgt._request_render()

	def _on_ambient_changed(self, value):
		"""ValSlider emits float directly."""
		tgt = self._get_tgt()
		if tgt:
			tgt.set_ambient(float(value))

	def _on_dir_intensity_changed(self, value):
		tgt = self._get_tgt()
		if tgt and hasattr(tgt, '_dir_light'):
			tgt._dir_light.intensity = float(value)
			tgt._request_render()

	def _on_point_h_changed(self, value):
		"""Point light horizontal angle changed."""
		tgt = self._get_tgt()
		if tgt:
			v_val = self.point_v_slider.getValue()
			tgt.set_light_position(value, v_val)

	def _on_point_v_changed(self, value):
		"""Point light vertical angle changed."""
		tgt = self._get_tgt()
		if tgt:
			h_val = self.point_h_slider.getValue()
			tgt.set_light_position(h_val, value)

	def _on_point_intensity_changed(self, value):
		"""Point light intensity changed."""
		tgt = self._get_tgt()
		if tgt:
			tgt.set_point_light_intensity(float(value))

	def _on_fov_changed(self, value):
		"""FOV changed - adjust distance proportionally to preserve apparent view size."""
		tgt = self._get_tgt()
		if not tgt or not hasattr(tgt, '_camera'):
			return
		old_fov = getattr(tgt._camera, 'fov', 50)
		new_fov = float(value)
		# Preserve apparent framing: new_dist = old_dist * old_fov / new_fov
		old_dist = tgt._get_camera_distance()
		new_dist = old_dist * old_fov / max(new_fov, 0.1)
		tgt._camera.fov = new_fov
		tgt._set_camera_distance(new_dist)
		self.dist_slider.setValue(new_dist, quiet=1)

	def _on_distance_changed(self, value):
		"""Distance changed - move camera along its current look direction."""
		tgt = self._get_tgt()
		if not tgt:
			return
		tgt._set_camera_distance(float(value))

	def _on_near_changed(self, value):
		tgt = self._get_tgt()
		if tgt and tgt._camera:
			tgt._camera.clip_near = max(0.01, float(value))
			tgt._request_render()

	def _on_far_changed(self, value):
		tgt = self._get_tgt()
		if tgt and tgt._camera:
			tgt._camera.clip_far = float(value)
			tgt._request_render()

	def _on_bg_color(self):
		color = QtWidgets.QColorDialog.getColor(QtCore.Qt.GlobalColor.black, self)
		if color.isValid():
			r = color.red() / 255.0
			g = color.green() / 255.0
			b = color.blue() / 255.0
			tgt = self._get_tgt()
			if tgt:
				tgt.set_bg_color(r, g, b)

	def _on_snapshot(self):
		filename, _ = QtWidgets.QFileDialog.getSaveFileName(
			self, "Save Snapshot", "", "Images (*.png *.tiff *.jpeg)"
		)
		if filename:
			tgt = self._get_tgt()
			if tgt and tgt._renderer:
				try:
					data = tgt._renderer.save()
					with open(filename, "wb") as f:
						f.write(data)
				except Exception as e:
					print("Error saving snapshot:", e)

	def update_tree(self):
		"""Rebuild the tree widget from the scene graph."""
		tgt = self._get_tgt()
		if not tgt:
			return

		self.tree.blockSignals(True)
		self.tree.clear()

		root = tgt.get_scene_root()
		self._build_tree_item(root, None)

		self.tree.expandItem(self.tree.topLevelItem(0))
		self.tree.blockSignals(False)

	def _build_tree_item(self, node, parent_item):
		item = QtWidgets.QTreeWidgetItem()
		item.setText(0, node.label)
		item.setData(0, QtCore.Qt.UserRole, node)

		check_state = QtCore.Qt.CheckState.Checked if node.visible else QtCore.Qt.CheckState.Unchecked
		item.setCheckState(0, check_state)

		# Column 1: type info
		if isinstance(node, SceneRoot):
			item.setText(1, "Scene")
		elif isinstance(node, ShapeNode):
			item.setText(1, node.__class__.__name__)
		else:
			item.setText(1, node.__class__.__name__)

		if parent_item is None:
			self.tree.addTopLevelItem(item)
		else:
			parent_item.addChild(item)

		for child in node.children:
			self._build_tree_item(child, item)


# ---------------------------------------------------------------------------
# Dialog for adding nodes
# ---------------------------------------------------------------------------

class AddNodeDialog(QtWidgets.QDialog):
	"""Dialog to configure and add a new scene node."""

	def __init__(self, inspector, tree_item):
		super().__init__(inspector)
		self.inspector = inspector
		self.tree_item = tree_item
		self.setWindowTitle("Add Object")
		self.setMinimumWidth(280)

		vbox = QtWidgets.QVBoxLayout(self)

		frame = QtWidgets.QFrame()
		frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
		fvbox = QtWidgets.QVBoxLayout(frame)

		label = QtWidgets.QLabel("Object Type:")
		fvbox.addWidget(label)

		self.type_combo = QtWidgets.QComboBox()
		self.type_combo.addItems(["Cube", "Sphere", "Cylinder", "Cone", "Arrow", "Data", "IsoSurface", "Slice", "VolumeRender", "ScatterPlot"])
		fvbox.addWidget(self.type_combo)

		name_label = QtWidgets.QLabel("Name:")
		fvbox.addWidget(name_label)
		self.name_edit = QtWidgets.QLineEdit("")
		fvbox.addWidget(self.name_edit)

		pos_label = QtWidgets.QLabel("Position (x, y, z):")
		fvbox.addWidget(pos_label)

		pos_layout = QtWidgets.QHBoxLayout()
		self.pos_x = QtWidgets.QDoubleSpinBox()
		self.pos_y = QtWidgets.QDoubleSpinBox()
		self.pos_z = QtWidgets.QDoubleSpinBox()
		for sb in [self.pos_x, self.pos_y, self.pos_z]:
			sb.setRange(-100, 100)
			sb.setSingleStep(0.1)
			sb.setValue(0.0)
		pos_layout.addWidget(self.pos_x)
		pos_layout.addWidget(self.pos_y)
		pos_layout.addWidget(self.pos_z)
		fvbox.addLayout(pos_layout)

		scl_label = QtWidgets.QLabel("Scale:")
		fvbox.addWidget(scl_label)
		self.scale_spin = QtWidgets.QDoubleSpinBox()
		self.scale_spin.setRange(0.01, 50)
		self.scale_spin.setValue(1.0)
		self.scale_spin.setSingleStep(0.1)
		fvbox.addWidget(self.scale_spin)

		frame.setLayout(fvbox)
		vbox.addWidget(frame)

		# Scatter plot data file loader (shown only for ScatterPlot type)
		self.scatter_frame = QtWidgets.QFrame()
		self.scatter_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
		sf_layout = QtWidgets.QHBoxLayout(self.scatter_frame)
		self.scatter_file_label = QtWidgets.QLabel("Data File:")
		sf_layout.addWidget(self.scatter_file_label)
		self.scatter_file_edit = QtWidgets.QLineEdit()
		sf_layout.addWidget(self.scatter_file_edit)
		self.scatter_file_btn = QtWidgets.QPushButton("...")
		self.scatter_file_btn.setFixedWidth(30)
		sf_layout.addWidget(self.scatter_file_btn)
		self.scatter_frame.setVisible(False)
		vbox.addWidget(self.scatter_frame)

		btn_layout = QtWidgets.QHBoxLayout()
		ok_btn = QtWidgets.QPushButton("Add")
		cancel_btn = QtWidgets.QPushButton("Cancel")
		btn_layout.addWidget(ok_btn)
		btn_layout.addWidget(cancel_btn)
		vbox.addLayout(btn_layout)

		ok_btn.clicked.connect(self.accept)
		cancel_btn.clicked.connect(self.reject)
		self.type_combo.currentIndexChanged.connect(self._on_type_changed)

		self.scatter_file_btn.clicked.connect(self._on_scatter_browse)

	def _on_type_changed(self, idx):
		"""Show/hide scatter file controls based on selected type."""
		stype = self.type_combo.currentText()
		self.scatter_frame.setVisible(stype == "ScatterPlot")

	def _on_scatter_browse(self):
		"""Open dialog to load scatter plot data file."""
		filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
			None, "Load Scatter Data", "",
			"Text Files (*.txt *.csv *.dat);;All Files (*)")
		if filepath:
			self.scatter_file_edit.setText(filepath)

	def get_data(self):
		result = {
			"type": self.type_combo.currentText(),
			"name": self.name_edit.text() or None,
			"position": (self.pos_x.value(), self.pos_y.value(), self.pos_z.value()),
			"scale": self.scale_spin.value(),
			"color": (0.8, 0.4, 0.2, 1.0),
		}
		# Load scatter data from file if ScatterPlot and file specified
		if result["type"] == "ScatterPlot":
			filepath = self.scatter_file_edit.text().strip()
			result["scatter_data"] = None
			if filepath:
				try:
					data = np.loadtxt(filepath)
					result["scatter_data"] = data.astype(np.float32)
				except Exception as e:
					print(f"Error loading scatter file: {e}")
		return result


# ---------------------------------------------------------------------------
# Test / main
# ---------------------------------------------------------------------------

def main():
	import sys
	from PySide6.QtWidgets import QApplication

	app = QApplication(sys.argv)

	widget = EMScene3DWidget()
	widget.resize(800, 600)
	widget.setWindowTitle("EMScene3D - 3D Scene Viewer")

	# Add some default objects
	widget.add_cube(name="Cube1", position=(-0.5, 0, 0), scale=0.5, color=(0.8, 0.4, 0.2, 1.0))
	widget.add_sphere(name="Sphere1", position=(0.5, 0, 0), scale=0.4, color=(0.4, 0.6, 0.8, 1.0))
	widget.add_cylinder(name="Cyl1", position=(0, 0, 0.5), scale=0.3, color=(0.6, 0.3, 0.7, 1.0))

	widget.show()
	print("Middle-click to show inspector.")
	print("Left-drag to orbit, scroll to zoom, right-drag to pan.")
	print("Keys: R=Rotate, T=Translate, S=Scale, Esc=Select, Del=Delete selected")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
