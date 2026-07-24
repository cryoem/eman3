#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ VtkIO)
# Copyright (c) 2000-2026 Baylor College of Medicine
#
# BSD/GPL license. See top-level for details.

"""VTK (Visualization Toolkit) Structured Points reader/writer.

Supports only STRUCTURED_POINTS datasets with FLOAT data type.
Binary VTK files store big-endian float32 pixel data after a text header.

Supported extensions: .vtk
"""

import os
import struct
import numpy as np


class FileFormatError(Exception):
    pass


class FileIOError(Exception):
    pass


class InvalidDimensions(Exception):
    pass


class VtkIO:
    """VTK Structured Points file reader/writer.

    Single 3D volume only. Writes BINARY big-endian float32 data.
    Reads both ASCII and BINARY formats but writes BINARY only.

    Parameters
    ----------
    filename : str
        Path to the .vtk file.
    mode : str, default "r"
        "r" for read-only, "rw" for write.
    """

    SUPPORT_STACK = False
    SUPPORT_3D = True
    SUPPORT_3D_STACK = False
    SUPPORTED_COMPRESS = False

    MAGIC = b"# vtk DataFile Version"

    def __init__(self, filename, mode="r"):
        self.filename = os.path.expanduser(filename)
        self.mode = mode

        _, ext = os.path.splitext(self.filename)
        if ext.lower() != ".vtk":
            raise FileFormatError(f"Unknown extension '{ext}' for VTK format.")

        self.nx = 0
        self.ny = 0
        self.nz = 0
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_z = 0.0
        self.spacing_x = 1.0
        self.spacing_y = 1.0
        self.spacing_z = 1.0

        try:
            if mode == "r":
                self._file = open(self.filename, "rb")
                self._parse_header()
            else:
                exists = os.path.exists(self.filename) and os.path.getsize(
                    self.filename) > 0
                mode_str = "r+b" if exists else "wb+"
                self._file = open(self.filename, mode_str)
                self._is_new_file = not exists
        except OSError as e:
            raise FileIOError(f"Cannot open file '{self.filename}': {e}")

    def __del__(self):
        if hasattr(self, "_file") and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

    @staticmethod
    def is_valid(filepath, chunk=None):
        """Check whether a file has a valid VTK header magic string."""
        try:
            if chunk is not None:
                return chunk.startswith(VtkIO.MAGIC)
            else:
                with open(os.path.expanduser(filepath), "rb") as f:
                    first = f.read(len(VtkIO.MAGIC))
                    return first == VtkIO.MAGIC
        except Exception:
            return False

    def _parse_header(self):
        """Parse the text header of a VTK Structured Points file."""
        self._file.seek(0)

        # Line 1: magic version line
        line = self._read_line()
        if not line.startswith(VtkIO.MAGIC.decode()):
            raise FileFormatError("Invalid VTK magic string")

        # Line 2: application name (ignored)
        _ = self._read_line()

        # Line 3: BINARY or ASCII
        filetype_line = self._read_line().strip().upper()
        if filetype_line not in ("BINARY", "ASCII"):
            raise FileFormatError(f"Unsupported VTK file type: {filetype_line}")
        self._is_binary = filetype_line == "BINARY"

        # Remaining lines: DATASET STRUCTURED_POINTS, DIMENSIONS, ORIGIN, SPACING,
        # then SCALARS and LOOKUP_TABLE before data begins
        self._data_offset = None
        while True:
            line = self._read_line()

            if line.startswith("DATASET"):
                dataset_type = line.split(maxsplit=1)[1].strip()
                if dataset_type != "STRUCTURED_POINTS":
                    raise FileFormatError(
                        f"Only STRUCTURED_POINTS supported, got {dataset_type}")
                # Next 3 lines: DIMENSIONS, ORIGIN, SPACING/ASPECT_RATIO
                for _ in range(3):
                    hdr_line = self._read_line()
                    if hdr_line.startswith("DIMENSIONS"):
                        parts = hdr_line.split()[1:]
                        self.nx = int(parts[0])
                        self.ny = int(parts[1])
                        self.nz = int(parts[2])
                    elif hdr_line.startswith("ORIGIN"):
                        parts = hdr_line.split()[1:]
                        self.origin_x = float(parts[0])
                        self.origin_y = float(parts[1])
                        self.origin_z = float(parts[2])
                    elif hdr_line.startswith("SPACING") or \
                         hdr_line.startswith("ASPECT_RATIO"):
                        parts = hdr_line.split()[1:]
                        self.spacing_x = float(parts[0])
                        self.spacing_y = float(parts[1])
                        self.spacing_z = float(parts[2])

            elif line.startswith("SCALARS"):
                # e.g. "SCALARS density float 1"
                parts = line.split()
                if len(parts) >= 3:
                    dtype_name = parts[2].lower()
                    if dtype_name not in ("float", "unsigned_short"):
                        raise FileFormatError(
                            f"Unsupported scalar type: {dtype_name}")

            elif line.startswith("LOOKUP_TABLE"):
                # Data begins after this line
                self._data_offset = self._file.tell()
                break

        if self._data_offset is None:
            raise FileFormatError("Could not find data offset in VTK header")

    def _read_line(self):
        """Read a single text line from the file."""
        raw = self._file.readline()
        return raw.decode("ascii", errors="replace").rstrip("\n\r")

    @staticmethod
    def bit_depths():
        return [0]  # float32 only for write

    def read_header(self, index=0):
        """Return header dict with nx, ny, nz, spacing, origin."""
        if index != 0:
            raise FileIOError("VTK is single-image; index must be 0")

        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "apix_x": self.spacing_x,
            "apiz_y": self.spacing_y,
            "apiz_z": self.spacing_z,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "origin_z": self.origin_z,
            "datatype": 0,  # float32
        }

    def read_data(self, index=0):
        """Read the VTK volume as a (nz, ny, nx) float32 array."""
        if index != 0:
            raise FileIOError("VTK is single-image; index must be 0")

        self._file.seek(self._data_offset)
        total = self.nx * self.ny * self.nz

        if self._is_binary:
            # VTK binary is big-endian float32
            raw = self._file.read(total * 4)
            data = np.frombuffer(raw, dtype=">f4").astype(np.float32)
        else:
            # ASCII — read remaining file content and parse whitespace-separated values
            ascii_text = self._file.read().decode("ascii", errors="replace")
            values = ascii_text.split()
            if len(values) != total:
                raise FileIOError(
                    f"ASCII VTK has {len(values)} values, expected {total}")
            data = np.array(values, dtype=np.float32)

        data = data.reshape((self.nz, self.ny, self.nx))
        return data

    def write_header(self, meta, index=-1):
        """Write a VTK Structured Points header.

        Returns 0 (single image). Only writes BINARY float format.
        """
        if self.mode == "r":
            raise FileIOError("VTK file opened read-only")
        if index != -1 and index != 0:
            raise FileIOError("VTK is single-image; index must be 0 or -1")

        self.nx = int(meta["nx"])
        self.ny = int(meta["ny"])
        self.nz = int(meta.get("nz", 1))
        self.origin_x = float(meta.get("origin_x", 0.0))
        self.origin_y = float(meta.get("origin_y", 0.0))
        self.origin_z = float(meta.get("origin_z", 0.0))
        self.spacing_x = float(meta.get("apix_x", 1.0))
        self.spacing_y = float(meta.get("apiz_y", 1.0))
        self.spacing_z = float(meta.get("apiz_z", 1.0))

        self._file.seek(0)
        self._file.truncate()

        f = self._file
        f.write(b"# vtk DataFile Version 2.0\n")
        f.write(b"EMAN3\n")
        f.write(b"BINARY\n")
        f.write(b"DATASET STRUCTURED_POINTS\n")
        f.write(
            f"DIMENSIONS {self.nx:d} {self.ny:d} {self.nz:d}\n"
            .encode())
        f.write(
            f"ORIGIN {self.origin_x:f} {self.origin_y:f} {self.origin_z:f}\n"
            .encode())
        f.write(
            f"SPACING {self.spacing_x:f} {self.spacing_y:f} {self.spacing_z:f}\n"
            .encode())

        npts = self.nx * self.ny * self.nz
        f.write(f"POINT_DATA {npts:d}\n".encode())
        f.write(b"SCALARS density float 1\n")
        f.write(b"LOOKUP_TABLE default\n")

        self._data_offset = f.tell()
        self._is_new_file = False

        return 0

    def write_data(self, data, index=0):
        """Write float32 volume data. VTK binary stores big-endian."""
        if self.mode == "r":
            raise FileIOError("VTK file opened read-only")
        if index != 0:
            raise FileIOError("VTK is single-image; index must be 0")

        data = np.asarray(data, dtype=np.float32)

        # Accept (ny,nx) for nz=1
        if self.nz == 1 and data.ndim == 2:
            data = data[np.newaxis, :]

        if data.shape != (self.nz, self.ny, self.nx):
            raise InvalidDimensions(
                f"Data shape {data.shape} doesn't match "
                f"({self.nz},{self.ny},{self.nx})")

        # VTK binary format is big-endian
        be_data = data.astype(">f4").tobytes()

        self._file.seek(self._data_offset)
        self._file.write(be_data)

    def flush(self):
        if self._file is not None:
            self._file.flush()

    def __len__(self):
        """Return 1 — VTK stores a single 3D volume."""
        return 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._file is not None:
                self._file.close()
                self._file = None
        except Exception:
            pass
        return False
