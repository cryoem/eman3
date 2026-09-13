#!/usr/bin/env python
"""IO handler for EMAN .lst (LSX) fast list files.

A .lst file is a list of records. Each record references an image (by index)
in another, actual image file, plus a JSON dictionary of parameters. When a
record is read, the referenced image is loaded and the JSON dictionary values
override the referenced image's header.

This handler therefore delegates the actual image I/O to the handler for the
referenced file, adding only record lookup and the header-override semantics.
It is read-only; use EMAN3.EMAN3.LSXFile directly to write .lst files.
"""

import os

from EMAN3.io.imageio import FileIOError


class LstIO:
    def __init__(self, filename, mode="r"):
        if not os.path.isfile(filename):
            raise FileIOError(f"LstIO: {filename} does not exist (use LSXFile to create .lst files)")
        # deferred import to avoid a circular import at module load
        from EMAN3.EMAN3 import LSXFile
        self._filename = filename
        self._mode = mode
        self._lsx = LSXFile(filename)

    @staticmethod
    def is_valid(filepath, chunk=None):
        if chunk is not None:
            return chunk.lstrip()[:4] == b"#LSX"
        try:
            with open(filepath, "rb") as f:
                return f.read(4).lstrip()[:4] == b"#LSX"
        except OSError:
            return False

    @property
    def nimg(self):
        return len(self._lsx)

    def __len__(self):
        return self.nimg

    def _record(self, index):
        if index < 0:
            index += self.nimg
        if index < 0 or index >= self.nimg:
            raise FileIOError(f"LstIO: image index {index} out of range (0-{self.nimg - 1})")
        rec = self._lsx.read(index)
        return rec[0], rec[1], rec[2]		# index in referenced file, referenced file, json dict

    def _override_header(self, hdr, jdict):
        if isinstance(jdict, dict):
            for k, v in jdict.items():
                if k != "lst_comment":
                    hdr[k] = v
        return hdr

    def read_header(self, index):
        from EMAN3.io.imageio import ImageIO
        idx_in_file, extfile, jdict = self._record(index)
        io = ImageIO(extfile)
        try:
            hdr = io.read_header(idx_in_file)
        finally:
            io.close()
        return self._override_header(hdr, jdict)

    def read_data(self, index):
        from EMAN3.io.imageio import ImageIO
        idx_in_file, extfile, jdict = self._record(index)
        io = ImageIO(extfile)
        try:
            data, hdr = io.read_image(idx_in_file)
        finally:
            io.close()
        return data

    def read_image(self, index):
        from EMAN3.io.imageio import ImageIO
        idx_in_file, extfile, jdict = self._record(index)
        io = ImageIO(extfile)
        try:
            data, hdr = io.read_image(idx_in_file)
        finally:
            io.close()
        return data, self._override_header(hdr, jdict)

    def write_header(self, meta, index=-1):
        raise NotImplementedError("LstIO is read-only; use EMAN3.EMAN3.LSXFile to write .lst records")

    def write_data(self, data, index=0):
        raise NotImplementedError("LstIO is read-only; use EMAN3.EMAN3.LSXFile to write .lst records")

    def close(self):
        if self._lsx is not None:
            try:
                self._lsx.close()
            except Exception:
                pass
            self._lsx = None
