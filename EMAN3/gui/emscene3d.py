#!/usr/bin/env python
"""EMScene3D - 3D scene viewer with hierarchical scene graph using PyGfx."""

import numpy as np
import weakref
import math

import pygfx as gfx
from rendercanvas.qt import QRenderWidget

from PySide6 import QtCore, QtGui, QtWidgets
from EMAN3.gui.valslider import ValSlider


# ---------------------------------------------------------------------------
# Scene node hierarchy (simplified, no EMAN2 deps)
# ---------------------------------------------------------------------------

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
			self._gfx_node.world.position = self._position

	def get_position(self):
		return self._position

	def set_scale(self, s):
		self._scale = (float(s), float(s), float(s))
		if self._gfx_node:
			self._gfx_node.world.scale = self._scale

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

		node = ShapeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		# Unit cube: 1x1x1 centered at origin (box_geometry creates -0.5..+0.5)
		geo = gfx.box_geometry(1.0, 1.0, 1.0)
		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		mesh = gfx.Mesh(geo, material)

		group = gfx.Group()
		group.add(mesh)
		group.world.position = position
		group.world.scale = (scale, scale, scale)
		node._gfx_node = group
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		node.label = label
		self._request_render()
		return node

	def add_sphere(self, name=None, position=(0, 0, 0), scale=1.0, color=(0.4, 0.6, 0.8, 1.0), parent=None):
		"""Add a sphere to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Sphere")

		node = ShapeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		# Unit sphere (radius 0.5 to fit in -1..+1 box)
		geo = gfx.sphere_geometry(0.5, width_segments=32, height_segments=24)
		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		mesh = gfx.Mesh(geo, material)

		group = gfx.Group()
		group.add(mesh)
		group.world.position = position
		group.world.scale = (scale, scale, scale)
		node._gfx_node = group
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		node.label = label
		self._request_render()
		return node

	def add_cylinder(self, name=None, position=(0, 0, 0), scale=1.0, color=(0.6, 0.3, 0.7, 1.0), parent=None):
		"""Add a cylinder to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Cylinder")

		node = ShapeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		# Cylinder: radius=0.5, height=1 (top/bottom at +/-0.5)
		geo = gfx.cylinder_geometry(radius_bottom=0.5, radius_top=0.5, height=1.0, radial_segments=32, height_segments=1)
		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		mesh = gfx.Mesh(geo, material)

		group = gfx.Group()
		group.add(mesh)
		group.world.position = position
		group.world.scale = (scale, scale, scale)
		node._gfx_node = group
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		node.label = label
		self._request_render()
		return node

	def add_cone(self, name=None, position=(0, 0, 0), scale=1.0, color=(0.9, 0.3, 0.3, 1.0), parent=None):
		"""Add a cone to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Cone")

		node = ShapeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		# Cone: radius=0.5, height=1 (centered)
		geo = gfx.cone_geometry(radius=0.5, height=1.0, radial_segments=32, open_ended=False)
		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		mesh = gfx.Mesh(geo, material)

		group = gfx.Group()
		group.add(mesh)
		group.world.position = position
		group.world.scale = (scale, scale, scale)
		node._gfx_node = group
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		node.label = label
		self._request_render()
		return node

	def add_line(self, name=None, start=(0, 0, 0), end=(1, 1, 1), color=(0.2, 1.0, 0.2, 1.0), parent=None):
		"""Add a line segment to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Line")

		node = ShapeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		points = np.array([start, end], dtype=np.float32)
		geo = gfx.Geometry(positions=points)
		material = gfx.LineMaterial(color=(color[0], color[1], color[2]), thickness=2.0)
		line = gfx.Line(geo, material)

		group = gfx.Group()
		group.add(line)
		node._gfx_node = group
		node.set_position(*start)

		self._root_group.add(group)
		node.label = label
		self._request_render()
		return node

	def add_arrow(self, name=None, position=(0, 0, 0), scale=1.0, color=(1.0, 0.2, 0.2, 1.0), parent=None):
		"""Add an arrow to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Arrow")

		node = ShapeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color

		# Arrow: use cone as head + cylinder as body merged
		head_geo = gfx.cone_geometry(radius=0.2, height=0.3, radial_segments=16, open_ended=True)
		body_geo = gfx.cylinder_geometry(radius_bottom=0.08, radius_top=0.08, height=0.7, radial_segments=12, height_segments=1)
		# Combine geometries by creating a merged Geometry
		import pygfx.utils as gfx_utils
		vs = np.vstack([head_geo.positions.data, body_geo.positions.data])
		is_combined = np.concatenate([head_geo.indices.data, body_geo.indices.data + len(head_geo.positions.data)])
		geo = gfx.Geometry(positions=vs, indices=is_combined)
		material = gfx.MeshStandardMaterial(color=(color[0], color[1], color[2]), roughness=0.6, metalness=0.1)
		mesh = gfx.Mesh(geo, material)

		group = gfx.Group()
		group.add(mesh)
		group.world.position = position
		group.world.scale = (scale, scale, scale)
		node._gfx_node = group
		node.set_position(*position)
		node.set_scale(scale)

		self._root_group.add(group)
		node.label = label
		self._request_render()
		return node

	def add_text(self, name=None, text="Hello", position=(0, 0, 0), scale=0.1, color=(1.0, 1.0, 1.0, 1.0), parent=None):
		"""Add 3D text to the scene."""
		if parent is None:
			parent = self._selected_node
		label = name or SceneNode.next_name("Text")

		node = ShapeNode(name=label, parent=parent)
		node.set_color(*color[:3])
		node._color = color
		node._text_content = text

		text_obj = gfx.Text(text=text, font_size=0.15, render_order=999)
		text_obj.material.color = (color[0], color[1], color[2])
		text_obj.local.position = position
		group = gfx.Group()
		group.add(text_obj)
		node._gfx_node = group
		node.set_position(*position)

		self._root_group.add(group)
		node.label = label
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
		layout = QtWidgets.QVBoxLayout(widget)

		# Tree widget showing scene hierarchy
		self.tree = QtWidgets.QTreeWidget()
		self.tree.setHeaderLabel("Scene Hierarchy")
		self.tree.setColumnCount(2)
		self.tree.setHeaderLabels(["Name", "Visible"])
		layout.addWidget(self.tree)

		# Buttons row
		btn_layout = QtWidgets.QHBoxLayout()
		self.add_btn = QtWidgets.QPushButton("Add Object")
		self.remove_btn = QtWidgets.QPushButton("Remove Object")
		btn_layout.addWidget(self.add_btn)
		btn_layout.addWidget(self.remove_btn)
		layout.addLayout(btn_layout)

		# Mouse mode info (OrbitController handles rotation/panning natively)
		info_label = QtWidgets.QLabel("Left-drag: Rotate | Right-drag: Pan | Scroll: Zoom")
		info_label.setStyleSheet("QLabel { color: gray; font-size: 9px; }")
		layout.addWidget(info_label)

		return widget


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
		self.type_combo.addItems(["Cube", "Sphere", "Cylinder", "Cone", "Arrow"])
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

		btn_layout = QtWidgets.QHBoxLayout()
		ok_btn = QtWidgets.QPushButton("Add")
		cancel_btn = QtWidgets.QPushButton("Cancel")
		btn_layout.addWidget(ok_btn)
		btn_layout.addWidget(cancel_btn)
		vbox.addLayout(btn_layout)

		ok_btn.clicked.connect(self.accept)
		cancel_btn.clicked.connect(self.reject)

	def get_data(self):
		return {
			"type": self.type_combo.currentText(),
			"name": self.name_edit.text() or None,
			"position": (self.pos_x.value(), self.pos_y.value(), self.pos_z.value()),
			"scale": self.scale_spin.value(),
			"color": (0.8, 0.4, 0.2, 1.0),
		}


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
