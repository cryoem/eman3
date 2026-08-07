#!/usr/bin/env python
"""Simple test application for the EMAN3 Multi-Image Matrix widget using PyGfx.

Usage:
    python e3x_test_imagemx.py [num_images]

If no argument given, generates 50 random 64x64 images.
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
	widget.resize(900, 700)

	n_images = int(sys.argv[1]) if len(sys.argv) >= 2 else 50
	img_size = 64

	np.random.seed(42)
	data = []
	for i in range(n_images):
		# Generate varied test images: some uniform, some with gradients, some noisy
		kind = (i // 4)%4
		if kind == 0:
			# Random noise
			img = np.random.rand(img_size, img_size).astype(np.float32)
		elif kind == 1:
			# Horizontal gradient
			yy, xx = np.mgrid[0:img_size, 0:img_size]
			img = (xx / img_size + 0.3 * np.random.rand(img_size, img_size)).astype(np.float32)
		elif kind == 2:
			# Circular pattern
			yy, xx = np.mgrid[0:img_size, 0:img_size]
			cx, cy = img_size / 2, img_size / 2
			r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
			img = (np.exp(-r/15) * (0.5 + 0.5*np.sin(r*0.5))).astype(np.float32)
		else:
			# Checkerboard with noise
			yy, xx = np.mgrid[0:img_size, 0:img_size]
			checker = ((xx // 8 + yy // 8) % 2).astype(np.float32)
			img = (checker * 0.7 + 0.15 + 0.15*np.random.rand(img_size, img_size)).astype(np.float32)

		data.append(img)

	widget.set_data(data, filename="test_matrix")
	widget.setWindowTitle(f"EMImageMX Test - {n_images} images ({img_size}x{img_size})")

	# Create a couple of test sets
	widget.enable_set("gradients", {i for i in range(n_images) if (i//4) % 4 == 1}, display=True)
	widget.enable_set("circles", {i for i in range(n_images) if (i//4) % 4 == 2}, display=True)
	widget.current_set = "gradients"

	widget.show()
	print(f"Loaded {n_images} test images ({img_size}x{img_size})")
	print("Middle-click to show inspector.")
	print("Scroll wheel to scroll through the grid.")
	print("Sets mode: click images to toggle set membership.")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
