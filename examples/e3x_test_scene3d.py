#!/usr/bin/env python
"""Test application for the EMAN3 3D Scene Viewer using PyGfx."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from PySide6.QtWidgets import QApplication
from EMAN3.gui.emscene3d import EMScene3DWidget


def main():
	app = QApplication(sys.argv)

	widget = EMScene3DWidget()
	widget.resize(900, 700)
	widget.setWindowTitle("EMScene3D Test")

	# Add some default objects to test rendering
	widget.add_cube(name="Red_Cube", position=(-1.0, 0.5, 0), scale=0.6,
		color=(0.9, 0.2, 0.2, 1.0))
	widget.add_sphere(name="Blue_Sphere", position=(1.0, -0.3, 0), scale=0.4,
		color=(0.2, 0.3, 0.9, 1.0))
	widget.add_cone(name="Green_Cone", position=(0, 1.0, 0), scale=0.5,
		color=(0.2, 0.8, 0.3, 1.0))

	# Add a cylinder and arrow
	widget.add_cylinder(name="Gray_Cyl",
		position=(-0.5, -0.8, 0.5), scale=0.3, color=(0.6, 0.6, 0.7, 1.0))
	widget.add_arrow(name="Orange_Arrow",
		position=(0.8, 0.8, -0.5), scale=0.4, color=(1.0, 0.5, 0.1, 1.0))

	# Add a line and text
	widget.add_line(name="Diagonal",
		start=(-1.5, -1.5, -1.5), end=(1.5, 1.5, 1.5), color=(1.0, 1.0, 1.0))
	widget.add_text(name="Label", text="Scene3D Test",
		position=(0, -1.2, 0), scale=0.12)

	# Generate sample 3D volume data (Gaussian blob + noise)
	size = 48
	zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
	cx, cy, cz = size / 2.0, size / 2.0, size / 2.0
	r = np.sqrt((xx - cx)**2 + (yy - cy)**2 + (zz - cz)**2)
	vol = np.exp(-r / 10.0).astype(np.float32) * 0.8
	vol += 0.15 * np.random.rand(size, size, size).astype(np.float32)

	# Add data node with volume and header metadata
	data_node = widget.add_data(name="Volume_Data",
		data=vol,
		header={"voxel_size": 1.0, "origin": (0, 0, 0)})

	# Add isosurface children - they inherit data from parent DataNode
	widget.add_isosurface(name="Iso_High", threshold=0.5,
		color=(1.0, 0.4, 0.2), parent=data_node)
	widget.add_isosurface(name="Iso_Low", threshold=0.2,
		color=(0.2, 0.6, 1.0), parent=data_node)

	widget.show()
	print("Middle-click to show inspector.")
	print("Left-drag to orbit, scroll to zoom, right-drag to pan.")
	print("Keys: R=Rotate, T=Translate, Esc=Select, Del=Delete selected")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
