#!/usr/bin/env python
#
# Author: Steven Ludtke (conversion from C++ EerIO)
# Copyright (c) 2020-2026 Baylor College of Medicine
#
# BSD/GPL license. See top-level for details.

"""EER (Event-Enabled Readout) file reader for direct electron detectors.

EER files are TIFF-based with custom RLE-encoded event data. Each frame
contains compressed event packets that must be decoded to produce an image.

Read-only format. Supports frame averaging and oversampling decoders:
  - I=0 (4Kx4K base resolution) — default
  - I=1 (8Kx8K, 2x oversampling)
  - I=2 (16Kx16K, 4x oversampling)

Frame averaging accumulates N raw frames into one averaged output image.

Supported extensions: .eer
"""

import os
import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


class FileFormatError(Exception):
    pass


class FileIOError(Exception):
    pass


class InvalidDimensions(Exception):
    pass


# ---------------------------------------------------------------------------
# XML metadata parsing helper (from C++ parse_acquisition_data)
# ---------------------------------------------------------------------------

def _to_snake_case(s):
    """Convert CamelCase to snake_case."""
    result = []
    for i, c in enumerate(s):
        if c.isupper():
            if i > 0 and not s[i - 1].isupper():
                result.append("_")
            result.append(c.lower())
        else:
            result.append(c)
    return "".join(result)


def _parse_xml_metadata(xml_str):
    """Parse EER acquisition metadata XML into a dict with 'EER.' prefix."""
    try:
        from xml.etree import ElementTree
        root = ElementTree.fromstring(xml_str)
    except Exception:
        return {}

    result = {}
    for item in root.findall("item"):
        name = item.get("name", "")
        value = item.text or ""
        key = "EER." + _to_snake_case(name)
        result[key] = value
    return result


# ---------------------------------------------------------------------------
# EER BitStream decoder — mirrors C++ BitStream<uint64_t>
# ---------------------------------------------------------------------------

class _BitStream:
    """Read bits from an array of uint64 little-endian words.

    Matches the C++ BitStream<uint64_t>: reads N bits from LSB side,
    advances through 64-bit words as needed.
    """
    __slots__ = ("words", "word_idx", "cur", "bit_counter")

    def __init__(self, data_bytes):
        self.words = np.frombuffer(data_bytes, dtype="<u8")
        self.word_idx = 0
        self.cur = int(self.words[0])
        self.bit_counter = 64

    def get_bits(self, n):
        """Extract N bits from the LSB of the current word."""
        result = self.cur & ((1 << n) - 1)

        if n < self.bit_counter:
            self.cur >>= n
            self.bit_counter -= n
        else:
            remaining = n - self.bit_counter
            self.word_idx += 1
            self.cur = int(self.words[self.word_idx])
            result |= (self.cur & ((1 << remaining) - 1)) << self.bit_counter
            self.cur >>= remaining
            self.bit_counter = 64 - remaining

        return result


def _read_rle(stream):
    """Read a 7-bit RLE value with overflow handling (max_val=127)."""
    val = 0
    while True:
        chunk = stream.get_bits(7)
        val += chunk
        if chunk != 127:
            break
    return val


def _read_subpix(stream):
    """Read a 4-bit sub-pixel position (no overflow)."""
    return stream.get_bits(4)


# ---------------------------------------------------------------------------
# Decoder — maps (count, sub_pix) -> (x, y) pixel coordinates
# ---------------------------------------------------------------------------

_CAMERA_SIZE = 4096
_CAM_BITS = 12


def _decode_coords(count, sub_pix, oversample_i):
    """Decode a single event to (x, y) coordinates."""
    base_x = count & (_CAMERA_SIZE - 1)
    base_y = count >> _CAM_BITS

    if oversample_i == 0:
        return base_x, base_y

    sub_x_bits = (sub_pix & 3) ^ 2
    sub_y_bits = (sub_pix >> 2) ^ 2
    shift = 2 - oversample_i

    x = (base_x << oversample_i) | (sub_x_bits >> shift)
    y = (base_y << oversample_i) | (sub_y_bits >> shift)
    return x, y


def _num_pix(oversample_i):
    """Output image dimension for a given oversampling level."""
    return _CAMERA_SIZE * (1 << oversample_i)


# ---------------------------------------------------------------------------
# Numba-JIT decoder (near-C throughput) — bit-exact mirror of the C++
# BitStream<uint64_t> / EerRle / EerSubPix / Decoder logic. Used when numba
# is available; the pure-Python decoder below remains as a fallback.
# ---------------------------------------------------------------------------

if _HAVE_NUMBA:
    @njit(cache=True)
    def _eer_get_bits(cur, bc, word_idx, words, n):
        """Read *n* bits from the LSB of *cur* (LSB-first, matching C++ BitStream).

        Returns (cur, bc, word_idx, bits, exhausted). Advances into *words*
        as needed; exhausted=True (bits=0) if the word supply runs out.
        """
        mask = (np.uint64(1) << n) - np.uint64(1)
        bits = cur & mask
        if n < bc:
            cur = cur >> n
            bc = bc - n
            return cur, bc, word_idx, bits, False
        remaining = n - bc
        word_idx = word_idx + 1
        if word_idx >= words.shape[0]:
            return cur, bc, word_idx, np.uint64(0), True
        cur = words[word_idx]
        bits = bits | ((cur & ((np.uint64(1) << remaining) - np.uint64(1))) << bc)
        cur = cur >> remaining
        bc = 64 - remaining
        return cur, bc, word_idx, bits, False

    @njit(cache=True)
    def _decode_frame_numba(words, max_count, oversample_i):
        """Decode compressed EER event words into a 2-D float32 image (numba JIT).

        *words* is a uint64 array of the raw frame bitstream. Semantics match the
        C++ decode_eer_data() exactly: 64-bit words read LSB-first, 7-bit RLE
        chunks (127 = continue), 4-bit sub-pixel nibble, and the DecoderIx<I>
        x/y mapping. If the bitstream is exhausted before max_count events the
        remaining pixels stay at 0.
        """
        pix = _CAMERA_SIZE * (1 << oversample_i)
        image = np.zeros((pix, pix), dtype=np.float32)
        if words.shape[0] == 0:
            return image

        cam_mask = _CAMERA_SIZE - 1
        cur = words[0]
        bc = 64
        word_idx = 0
        exhausted = False

        # initial RLE value (7-bit chunks, 127 = overflow)
        rle_val = np.uint64(0)
        while True:
            cur, bc, word_idx, chunk, exhausted = _eer_get_bits(cur, bc, word_idx, words, 7)
            if exhausted:
                return image
            rle_val = rle_val + chunk
            if chunk != 127:
                break
        # initial sub-pixel nibble
        cur, bc, word_idx, sub_pix, exhausted = _eer_get_bits(cur, bc, word_idx, words, 4)
        if exhausted:
            return image
        count = np.int64(rle_val)

        while count < max_count:
            if oversample_i == 0:
                x = count & cam_mask
                y = count >> _CAM_BITS
            else:
                bx = count & cam_mask
                by = count >> _CAM_BITS
                x = (bx << oversample_i) | (((sub_pix & np.uint64(3)) ^ np.uint64(2)) >> (2 - oversample_i))
                y = (by << oversample_i) | (((sub_pix >> np.uint64(2)) ^ np.uint64(2)) >> (2 - oversample_i))
            image[y, x] += 1

            # next RLE value
            rle_val = np.uint64(0)
            exhausted = False
            while True:
                cur, bc, word_idx, chunk, exhausted = _eer_get_bits(cur, bc, word_idx, words, 7)
                if exhausted:
                    break
                rle_val = rle_val + chunk
                if chunk != 127:
                    break
            if exhausted:
                break
            # next sub-pixel nibble
            cur, bc, word_idx, sub_pix, exhausted = _eer_get_bits(cur, bc, word_idx, words, 4)
            if exhausted:
                break
            count = count + np.int64(rle_val) + 1

        return image


# ---------------------------------------------------------------------------
# Frame decoding — decompress one EER frame into an image array
# ---------------------------------------------------------------------------

def _decode_frame(raw_data, max_count, oversample_i):
    """Decode compressed EER event data into a 2-D float32 image.

    Mirrors the C++ decode_eer_data() logic:
      1) Read first RLE + SubPix -> initial count = rle value
      2) While count < max_count: add coordinate, read next RLE+SubPix, count += rle+1
      3) If bitstream runs out before max_count, remaining pixels stay at 0.

    Uses the numba-JIT implementation when available (near-C throughput);
    otherwise falls back to the pure-Python decoder below.
    """
    if _HAVE_NUMBA:
        words = np.frombuffer(raw_data, dtype="<u8")
        return _decode_frame_numba(words, max_count, oversample_i)

    pix = _num_pix(oversample_i)
    image = np.zeros((pix, pix), dtype=np.float32)

    stream = _BitStream(raw_data)

    rle_val = _read_rle(stream)
    sub_pix = _read_subpix(stream)
    count = rle_val

    try:
        while count < max_count:
            x, y = _decode_coords(count, sub_pix, oversample_i)
            image[y, x] += 1

            rle_val = _read_rle(stream)
            sub_pix = _read_subpix(stream)
            count += rle_val + 1
    except IndexError:
        # Bitstream exhausted before reaching max_count — remaining pixels
        # stay at 0, matching C++ behavior (output initialized to zero).
        pass

    return image


# ---------------------------------------------------------------------------
# EerIO class — uses tifffile for TIFF I/O (avoids ctypes libtiff segfaults)
# ---------------------------------------------------------------------------

class EerIO:
    """EER (Event-Enabled Readout) file reader.

    Parameters
    ----------
    filename : str
        Path to the .eer file.
    frame_averaging : int, default 1
        Number of raw frames to average into one output image.
    oversample : int, default 0
        Oversampling level: 0=4K, 1=8K, 2=16K output dimensions.

    Attributes
    ----------
    nimg : int
        Number of averaged frames available (= raw_frames // frame_averaging).
    """

    SUPPORT_STACK = True
    SUPPORT_3D = False
    SUPPORT_3D_STACK = False
    SUPPORTED_COMPRESS = False

    def __init__(self, filename, frame_averaging=1, oversample=0):
        self.filename = os.path.expanduser(filename)

        _, ext = os.path.splitext(self.filename)
        if ext.lower() != ".eer":
            raise FileFormatError(f"Unknown extension '{ext}' for EER format.")

        self._oversample_i = int(oversample)
        if self._oversample_i not in (0, 1, 2):
            raise ValueError("oversample must be 0, 1, or 2")

        self._frame_averaging = max(1, int(frame_averaging))

        # Open with tifffile for reliable frame counting and raw access
        import tifffile
        self._tif = tifffile.TiffFile(self.filename)

        # Count raw frames from tifffile pages
        self._raw_num_frames = len(self._tif.pages)
        self.nimg = self._raw_num_frames // self._frame_averaging
        self._pix_dim = _num_pix(self._oversample_i)

        # Precompute the (dataoffset, bytecount) strip table for every raw
        # frame so per-frame reads never have to touch tifffile pages again.
        self._frame_strips = [
            (list(page.dataoffsets), list(page.databytecounts))
            for page in self._tif.pages
        ]

        # One persistent binary handle for raw strip reads (open/close per
        # frame is expensive); closed in __del__/__exit__.
        self._file = open(self.filename, "rb")

        # Parse acquisition metadata from XML tag (65001) on first page
        self._acquisition_data = {}
        try:
            xml_tags = self._tif.pages[0].tags.get(65001)
            if xml_tags:
                xml_bytes = xml_tags.value
                if isinstance(xml_bytes, bytes):
                    self._acquisition_data = _parse_xml_metadata(
                        xml_bytes.decode("utf-8", errors="replace"))
        except Exception:
            pass

    def close(self):
        """Close the persistent raw-read handle and the TIFF accessor."""
        try:
            if hasattr(self, "_file") and self._file is not None:
                self._file.close()
                self._file = None
        except Exception:
            pass
        try:
            if hasattr(self, "_tif") and self._tif is not None:
                self._tif.close()
                self._tif = None
        except Exception:
            pass

    def __del__(self):
        self.close()

    @staticmethod
    def is_valid(filepath, chunk=None):
        """Check whether a file is a valid EER file by trying to open it as TIFF.
        If chunk is provided, checks for TIFF magic bytes in the chunk.
        """
        try:
            if chunk is not None:
                # TIFF magic: II (LE) or MM (BE), offset 2 = 42 or 43
                import struct
                if len(chunk) < 4:
                    return False
                if chunk[:2] in (b'II', b'MM'):
                    offset = struct.unpack('<H' if chunk[:2] == b'II' else '>H', chunk[2:4])[0]
                    return offset in (42, 43)
                return False
            import tifffile
            with tifffile.TiffFile(os.path.expanduser(filepath)) as tif:
                return len(tif.pages) > 0
        except Exception:
            return False

    def _read_raw_frame(self, frame_idx):
        """Seek to *frame_idx* and read raw compressed strip data."""
        if frame_idx >= self._raw_num_frames:
            raise FileIOError(
                f"Frame index {frame_idx} out of range (0..{self._raw_num_frames - 1})")

        # Read raw compressed bytes from the TIFF file using the precomputed
        # strip offsets, over the single persistent binary handle.
        offsets, counts = self._frame_strips[frame_idx]
        all_data = bytearray()
        f = self._file
        for offset, count in zip(offsets, counts):
            f.seek(offset)
            all_data.extend(f.read(count))

        return bytes(all_data)

    def read_header(self, index=0):
        """Return header dict for the averaged frame at *index*."""
        if index < 0:
            raise ValueError("EerIO does not support appending (index < 0)")
        if index >= self.nimg:
            raise FileIOError(
                f"Frame index {index} out of range (0..{self.nimg - 1})")

        raw_idx = index * self._frame_averaging
        # Verify accessibility by seeking to the frame
        try:
            _ = self._tif.pages[raw_idx]
        except IndexError:
            raise FileIOError(f"Cannot seek to frame {raw_idx}")

        nx = self._pix_dim
        ny = self._pix_dim
        apix_x = 1.0
        apix_y = 1.0

        hdr = {"nx": nx, "ny": ny, "nz": 1}
        hdr.update(self._acquisition_data)

        spw = self._acquisition_data.get("EER.sensor_pixel_size.width")
        sph = self._acquisition_data.get("EER.sensor_pixel_size.height")
        if spw:
            try:
                val = float(spw)
                apix_x = val * 1.0e10 if val != 0 else 1.0
            except (ValueError, TypeError):
                pass
        if sph:
            try:
                val = float(sph)
                apix_y = val * 1.0e10 if val != 0 else 1.0
            except (ValueError, TypeError):
                pass

        hdr["apix_x"] = apix_x
        hdr["apix_y"] = apix_y
        hdr["apix_z"] = apix_x
        return hdr

    def read_data(self, index=0):
        """Read and decode the averaged frame at *index*."""
        if index < 0:
            raise ValueError("EerIO does not support negative index")
        if index >= self.nimg:
            raise FileIOError(
                f"Frame index {index} out of range (0..{self.nimg - 1})")

        raw_start = index * self._frame_averaging
        raw_end = min(raw_start + self._frame_averaging, self._raw_num_frames)

        pix = self._pix_dim
        accumulator = np.zeros((pix, pix), dtype=np.float32)

        for raw_idx in range(raw_start, raw_end):
            raw_data = self._read_raw_frame(raw_idx)
            frame = _decode_frame(
                raw_data, _CAMERA_SIZE * _CAMERA_SIZE, self._oversample_i)
            accumulator += frame

        n_frames = raw_end - raw_start
        if n_frames > 0:
            accumulator /= n_frames

        return accumulator

    def write_header(self, d, index=0):
        raise FileIOError("EER format is read-only; writing not supported.")

    def write_data(self, data, index=0):
        raise FileIOError("EER format is read-only; writing not supported.")

    def flush(self):
        pass

    @staticmethod
    def bit_depths():
        return [32]

    def __len__(self):
        """Return number of images in the file."""
        return self.nimg

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
