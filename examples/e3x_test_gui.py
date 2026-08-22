#!/usr/bin/env python
"""Simple test application for the EMAN3 GUI widget using PyGfx rendering.

Usage:
    python test_gui.py <image_filename>

Loads the given image file into a 2D viewer window with full navigation,
contrast controls, and shape overlay support via PyGfx.
"""

import faulthandler
faulthandler.enable()

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

print("Python segfault handler enabled.")

from PySide6.QtWidgets import QApplication

from EMAN3.io.imageio import ImageIO
from EMAN3.gui.emimage2d import EMImage2DWidget


def main():
	if len(sys.argv) < 2:
		print("Usage: python test_gui.py <image_filename>")
		sys.exit(1)

	image_path = sys.argv[1]
	if not os.path.exists(image_path):
		print(f"Error: File '{image_path}' not found.")
		sys.exit(1)

	app = QApplication(sys.argv)

	# Load image using ImageIO wrapper
	try:
		io = ImageIO(image_path, "r")
	except Exception as e:
		print(f"Error opening file: {e}")
		sys.exit(1)

	nimg = io.nimg
	print(f"Loaded: {image_path} ({nimg} image(s))")

	# Read first image (or all if stack)
	if nimg > 1:
		data, headers = io.read_images()
		print(f"Read stack: {data.shape}, {len(headers)} headers")
		# Convert (N, ny, nx) to list of 2D slices for the widget
		stack_data = [data[i] for i in range(data.shape[0])]
	else:
		data, header = io.read_image(0)
		print(f"Read image: {data.shape}, header keys: {list(header.keys())}")
		stack_data = data if data.ndim == 2 else [data[i] for i in range(data.shape[0])]

	# Create the viewer widget with appropriate size
	widget = EMImage2DWidget(stack=stack_data, metadata=header if nimg == 1 else headers)
	widget.setWindowTitle(f"EMAN3 Viewer - {os.path.basename(image_path)}")

	# Get dimensions from first image in stack
	first_img = stack_data[0] if isinstance(stack_data, list) else stack_data
	if first_img.ndim == 2:
		ny, nx = first_img.shape
	else:
		nz, ny, nx = first_img.shape

	# Set initial window size based on image dimensions (scale to fit screen)
	window_w = min(nx * 3, 1920)
	window_h = min(ny * 3, 1080)
	widget.resize(max(window_w, 512), max(window_h, 512))

	widget.show()
	print("Middle-click or Alt+click on the display to show/hide control panel.")
	print("Right-drag to pan, scroll wheel to zoom.")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
