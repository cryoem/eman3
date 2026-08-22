#!/usr/bin/env python
"""Simple test application for the EMAN3 3D Plot widget using PyGfx."""

import sys
import os
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from PySide6.QtWidgets import QApplication
from EMAN3.gui.emplot3d import EMPlot3DWidget


def main():
	app = QApplication(sys.argv)

	widget = EMPlot3DWidget()
	widget.resize(800, 700)

	if len(sys.argv) >= 2:
		for i, data_path in enumerate(sys.argv[1:], start=1):
			try:
				raw = np.loadtxt(data_path)
				if raw.ndim == 1:
					x = np.arange(len(raw)).astype(np.float32)
					y = raw.astype(np.float32)
					z = y.copy()
					cols = [x, y, z]
				elif raw.ndim == 2 and raw.shape[1] >= 3:
					cols = [raw[:, j].astype(np.float32) for j in range(raw.shape[1])]
				else:
					print(f"Skipping '{data_path}': need at least 3 columns")
					continue

				key = os.path.relpath(data_path)
				widget.set_data(cols, key=key, replace=(i == 1))
				print(f"Loaded: {data_path} as '{key}' ({raw.shape[0]} rows, {len(cols)} columns)")

			except Exception as e:
				traceback.print_exc()
				print(f"Error loading file: {e}")
	else:
		# Generate sample 3D data
		n = 500
		t = np.linspace(0, 20 * np.pi, n).astype(np.float32)

		x = (np.cos(t) + 0.5 * np.sin(3 * t)).astype(np.float32)
		y = (np.sin(t) - 0.5 * np.sin(2 * t)).astype(np.float32)
		z = (t / (20 * np.pi) * 4 - 2).astype(np.float32)

		widget.set_data([x, y, z], key="Lissajous spiral", replace=True)
		print("Generated sample data: Lissajous spiral")

	widget.set_axis_parms("X", "Y", "Z")

	widget.show()
	print("Press 'R' to rescale, 'C' for controls.")
	print("Left-drag to rotate (OrbitController), scroll to zoom.")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
