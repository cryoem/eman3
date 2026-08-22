#!/usr/bin/env python
"""Simple test application for the EMAN3 Multi-Image Matrix widget using PyGfx.

Usage:
    e3x_test_imagemx [image_stack_file]

If no file argument is provided, generates 50 random test images (64x64).
Otherwise loads the given image stack as a tiled grid of 2D slices.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from PySide6.QtWidgets import QApplication
from EMAN3.gui.emimagemx import EMImageMXWidget


def main():
	app = QApplication(sys.argv)

	widget = EMImageMXWidget()
	widget.resize(800, 700)

	if len(sys.argv) >= 2 and os.path.isfile(sys.argv[1]):
		# Load from image file
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
				# If 3D, take slices as list; otherwise single image
				if data.ndim == 3:
					stack_data = [data[i] for i in range(data.shape[0])]
					headers = [header] * len(stack_data)
				else:
					stack_data = [data]
					headers = [header]

			widget.set_data(stack_data, filename=image_path)
			# Pass headers to the widget after data is loaded
			if hasattr(headers, '__iter__'):
				widget._data_headers = list(headers)
		except Exception as e:
			import traceback
			traceback.print_exc()
			print(f"Error loading file: {e}")
			return
	else:
		# Generate test data: random 64x64 images
		n_images = int(sys.argv[1]) if len(sys.argv) >= 2 else 50
		img_size = 64

		np.random.seed(42)
		data = []
		for i in range(n_images):
			kind = i % 4
			if kind == 0:
				img = np.random.rand(img_size, img_size).astype(np.float32)
			elif kind == 1:
				yy, xx = np.mgrid[0:img_size, 0:img_size]
				img = (xx / img_size + 0.3 * np.random.rand(img_size, img_size)).astype(np.float32)
			elif kind == 2:
				yy, xx = np.mgrid[0:img_size, 0:img_size]
				cx, cy = img_size / 2, img_size / 2
				r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
				img = (np.exp(-r/15) * (0.5 + 0.5*np.sin(r*0.5))).astype(np.float32)
			else:
				yy, xx = np.mgrid[0:img_size, 0:img_size]
				checker = ((xx // 8 + yy // 8) % 2).astype(np.float32)
				img = (checker * 0.7 + 0.15 + 0.15*np.random.rand(img_size, img_size)).astype(np.float32)

			data.append(img)

			widget.set_data(data, filename="test_matrix")
			# Add sample headers so Values dropdown works for generated test data
			widget._data_headers = [
				{"kind": {0: "random", 1: "gradient", 2: "circle", 3: "checker"}[i % 4],
				 "index": i}
				for i in range(n_images)
			]
		widget.setWindowTitle(f"EMImageMX Test - {n_images} images ({img_size}x{img_size})")

		# Create a couple of test sets
		widget.enable_set("gradients", {i for i in range(n_images) if i % 4 == 1}, display=True)
		widget.enable_set("circles", {i for i in range(n_images) if i % 4 == 2}, display=True)
		widget.current_set = "gradients"

	widget.show()
	print("Middle-click to show inspector.")
	print("Scroll wheel to adjust tile scale.")
	print("Right-drag to scroll the grid vertically.")
	print("Sets mode: click images to toggle set membership.")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
