#!/usr/bin/env python
#
# e3display.py (EMAN3)
# Steve Ludtke
#
# Display EMAN3 data files in GUI windows.
#
# With no arguments, opens the EMAN3 file browser.
#
# Given a list of files (and no action option), each file is displayed with
# its default browser action (the action triggered by a double-click in the
# browser); no browser window itself is opened.
#
# A mutually exclusive action option may be given to apply one specific
# browser action to all of the listed files instead (no mixed operations):
#
#   --image2d     show each file in its own 2-D image window
#   --image3d     show all files in a single 3-D scene window
#   --imagemx     show each file's images in its own grid window
#   --plot2d      plot all files in a single 2-D plot window
#   --plot3d      plot all files in a single 3-D plot window
#   --histogram   histogram all files in a single histogram window
#
# Only files compatible with the chosen action (as in the browser) may be
# listed.
#
import argparse
import os
import sys

# Option -> the browser button name(s) marking a file as compatible with it.
# (The first name present in the file's action list is used.)
ACTION_BUTTONS = {
	"image2d": ("Show 2D+",),
	"image3d": ("Show 3D", "Show all 3D"),
	"imagemx": ("Show Stack+",),
	"plot2d": ("Plot 2D",),
	"plot3d": ("Plot 3D",),
	"histogram": ("Hist",),
}


def parse_args(argv):
	parser = argparse.ArgumentParser(
		prog="e3display",
		description="Display EMAN3 files in GUI windows. With no arguments, "
		"opens the file browser; with a file list, applies each file's default "
		"browser action (or the one selected with an --option) without opening "
		"a browser window.")
	parser.add_argument("files", nargs="*", help="files to display")
	grp = parser.add_mutually_exclusive_group()
	grp.add_argument("--image2d", action="store_true",
		help="show each file in its own 2-D image window")
	grp.add_argument("--image3d", action="store_true",
		help="show all files in a single 3-D scene window")
	grp.add_argument("--imagemx", action="store_true",
		help="show each file's images in its own grid window")
	grp.add_argument("--plot2d", action="store_true",
		help="plot all files in a single 2-D plot window")
	grp.add_argument("--plot3d", action="store_true",
		help="plot all files in a single 3-D plot window")
	grp.add_argument("--histogram", action="store_true",
		help="histogram all files in a single histogram window")
	return parser.parse_args(argv)


def file_type_for(path):
	"""Resolve a file to its browser FileType instance, using the same
	classification as the browser metadata cache (e3browsercache)."""
	if not os.path.isfile(path):
		return None
	from EMAN3.gui.embrowser import EMFileType
	from programs.e3browsercache import classify
	try:
		meta = classify(path)
	except Exception:
		meta = None
	typ = meta.get("type") if meta else None
	cls = EMFileType.typesbyft.get(typ)
	if cls is None:
		return None
	ft = cls(path)
	if cls is EMFileType.typesbyft.get("Data") and getattr(ft, "ncol", 0) <= 2:
		# The Data action set depends on the column count, normally taken from
		# the browser cache; resolve it directly from the file when absent.
		try:
			cols, _labels = ft.plot_data()
			if cols:
				ft.ncol = len(cols)
		except Exception:
			pass
	return ft


def find_action(ft, names):
	"""Return the first action (name, tip, callback) of ft whose name is in
	'names', or None."""
	for a in ft.actions():
		if a[0] in names:
			return a
	return None


def show_file(brws, path, ft, option):
	"""Display one file: its default action, or the selected browser action."""
	if option is None:
		acts = ft.actions()
		if not acts:
			raise ValueError("no default display action")
		name, _tip, fn = acts[0]
		print("%s: %s" % (path, name))
		fn(brws)
		return

	action = find_action(ft, ACTION_BUTTONS[option])
	if action is None:
		raise ValueError("not compatible with --%s" % option)
	print("%s: %s" % (path, option))

	if option == "image3d":
		# All volumes of all files go into one window; invoke the shared
		# helper (new=False) rather than the button callback, since the
		# 3-D stack buttons always open new windows.
		brws.busy()
		try:
			brws._show_scene3d(brws._read_volumes(path), path, new=False)
		finally:
			brws.notbusy()
	elif option in ("plot2d", "plot3d", "histogram"):
		# All files accumulate in one window; invoke the shared helper
		# (new=False) uniformly, since not every file type's button callback
		# does so (e.g. a 1-D stack's Plot 2D button opens a new window).
		brws.busy()
		try:
			columns, labels = ft.plot_data()
			if option == "plot2d":
				brws._show_plot2d(columns, path, new=False, column_labels=labels)
			elif option == "plot3d":
				brws._show_plot3d(columns, path, new=False, column_labels=labels)
			else:
				brws._show_hist(columns, path, new=False, column_labels=labels)
		finally:
			brws.notbusy()
	else:
		# image2d / imagemx: the + button callbacks open a new window, giving
		# each file its own window as required.
		action[2](brws)


def main(argv=None):
	args = parse_args(argv if argv is not None else sys.argv[1:])

	from PySide6 import QtWidgets
	app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
	from EMAN3.gui.embrowser import EMBrowserWidget

	if not args.files:
		# No files: just open the browser.
		brws = EMBrowserWidget()
		brws.show()
		return app.exec()

	option = None
	for opt in ("image2d", "image3d", "imagemx", "plot2d", "plot3d", "histogram"):
		if getattr(args, opt):
			option = opt
			break

	# Headless browser: needed only for the display helpers. Stop the
	# metadata-cache timer so no background updater processes are launched.
	brws = EMBrowserWidget(startpath=".")
	brws._cache_timer.stop()
	brws._updaters.clear()

	nerrors = 0
	for f in args.files:
		path = os.path.abspath(os.path.expanduser(f))
		if not os.path.isfile(path):
			print("e3display: no such file: %s" % f, file=sys.stderr)
			nerrors += 1
			continue
		try:
			ft = file_type_for(path)
			if ft is None:
				print("e3display: %s: could not determine file type" % path, file=sys.stderr)
				nerrors += 1
				continue
			show_file(brws, path, ft, option)
		except Exception as e:
			print("e3display: %s: %s" % (path, e), file=sys.stderr)
			nerrors += 1

	if nerrors:
		# Still run the event loop so the successfully displayed windows stay
		# open; report failure via the exit code.
		rc = app.exec()
		return rc if rc else 1
	return app.exec()


if __name__ == "__main__":
	sys.exit(main())
