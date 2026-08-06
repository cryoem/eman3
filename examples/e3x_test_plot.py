#!/usr/bin/env python
"""Simple test application for the EMAN3 2D Plot widget using PyGfx.

Usage:
    python e3x_test_plot.py [data_file1] [data_file2] ...

If no files are given, generates sample data (sine/cosine curves).
Otherwise loads each file as a separate data set. Each file should have
at least 2 numeric columns; all columns are loaded so the inspector
can select any as X or Y.
"""

import sys
import os
import numpy as np
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from PySide6.QtWidgets import QApplication
from EMAN3.gui.emplot2d import EMPlot2DWidget


def main():
	app = QApplication(sys.argv)

	widget = EMPlot2DWidget()
	widget.resize(800, 600)

	if len(sys.argv) >= 2:
		# Load from file(s) - each file becomes a separate data set
		for i, data_path in enumerate(sys.argv[1:], start=1):
			try:
				raw = np.loadtxt(data_path)
				if raw.ndim == 1:
					x = np.arange(len(raw)).astype(np.float32)
					y = raw.astype(np.float32)
					cols = [x, y]
				elif raw.ndim == 2 and raw.shape[1] >= 2:
					# Load ALL columns so inspector can select any of them
					cols = [raw[:, j].astype(np.float32) for j in range(raw.shape[1])]
				else:
					print(f"Skipping '{data_path}': need at least 2 columns")
					continue

				key = os.path.relpath(data_path)
				widget.set_data(cols, key=key, replace=(i == 1))
				print(f"Loaded: {data_path} as '{key}' ({raw.shape[0]} rows, {len(cols)} columns)")

			except Exception as e:
				traceback.print_exc()
				print(f"Error loading file: {e}")
	else:
		# Generate sample data
		n = 200
		x = np.linspace(0, 4 * np.pi, n).astype(np.float32)
		
		# Sine wave
		y1 = np.sin(x).astype(np.float32)
		widget.set_data([x, y1], key="sin(x)", replace=True)
		
		# Cosine wave
		y2 = np.cos(x).astype(np.float32)
		widget.set_data([x, y2], key="cos(x)")
		
		# Damped oscillation
		y3 = np.exp(-x / (4 * np.pi)) * np.sin(3 * x).astype(np.float32)
		widget.set_data([x, y3], key="damped sin")
		
		print("Generated sample data: sin(x), cos(x), damped oscillation")
		
	# Set axis labels and title
	widget.set_axis_parms("X", "Y")

	widget.show()
	print("Middle-click or Alt+click to show/hide control panel.")
	print("Right-drag to pan, scroll wheel to zoom.")
	print("Press 'R' to rescale, 'C' for controls.")
	sys.exit(app.exec())


if __name__ == "__main__":
	main()
