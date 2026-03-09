"""UECP (Universal Encoder Communications Protocol) 6.02 frame parser.

Self-contained: no dependencies on app.py.

Frame wire format (between STA 0xFE and STP 0xFF, after byte de-stuffing):
    ADR(2) SEQ(1) MEL(1) MEM(MEL bytes) CRC(2)

Message Element wire format within MEM:
    MEC(1) [DSN(1) PSN(1)] [MEL_len(1)] data(...)

Byte stuffing on the wire:
    0xFD 0xDD  -> 0xFD
    0xFD 0xDE  -> 0xFE
    0xFD 0xDF  -> 0xFF

Reference: UECP specification SPB 490 version 6.02 (September 2006).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

# ---------------------------------------------------------------------------
# Frame delimiters
# ---------------------------------------------------------------------------

UECP_STA: int = 0xFE
UECP_STP: int = 0xFF
UECP_ESC: int = 0xFD

_STUFFED: dict[int, int] = {0xDD: 0xFD, 0xDE: 0xFE, 0xDF: 0xFF}

# ---------------------------------------------------------------------------
# Message Element Codes
# ---------------------------------------------------------------------------

MEC_PI:      int = 0x01   # Programme Identification
MEC_PS:      int = 0x02   # Programme Service name
MEC_TA_TP:   int = 0x03   # Traffic Announcement / Traffic Programme
MEC_DI_PTYI: int = 0x04   # Decoder Identification / Programme Type Indicator
MEC_MS:      int = 0x05   # Music / Speech switch
MEC_PIN:     int = 0x06   # Programme Item Number (Group 1A Block D)
MEC_PTY:     int = 0x07   # Programme Type
MEC_RT:      int = 0x0A   # RadioText (Group 2A)
MEC_AF:      int = 0x13   # Alternative Frequencies (Method A)
MEC_SLC:     int = 0x1A   # Slow Labelling Codes (Group 1A Block C)
MEC_FFG:     int = 0x24   # Free Format Group (freeform data into any RDS group)
MEC_ODA_SET:  int = 0x40   # ODA Set (assign ODA AID to a group, signal via Group 3A)
MEC_ODA_DATA: int = 0x46   # ODA Data (send group data for a registered ODA AID)

MEC_NAMES: dict[int, str] = {
    MEC_PI:      "PI",
    MEC_PS:      "PS",
    MEC_TA_TP:   "TA/TP",
    MEC_DI_PTYI: "DI/PTYI",
    MEC_MS:      "M/S",
    MEC_PIN:     "PIN",
    MEC_PTY:     "PTY",
    MEC_RT:      "RT",
    MEC_AF:      "AF",
    MEC_SLC:     "SLC",
    MEC_FFG:     "FFG",
    MEC_ODA_SET:  "ODA_SET",
    MEC_ODA_DATA: "ODA_DATA",
}

# MECs with a fixed data length (no MEL_len byte on the wire)
_FIXED_MEL: dict[int, int] = {
    MEC_PI:      2,
    MEC_PS:      8,
    MEC_TA_TP:   1,
    MEC_DI_PTYI: 1,
    MEC_MS:      1,
    MEC_PIN:     2,
    MEC_PTY:     1,
    MEC_SLC:     2,
}

# MECs with no DSN/PSN fields AND a fixed data length (no mel_len byte either)
_NO_ADDR_FIXED_MEL: dict[int, int] = {
    MEC_FFG:     6,  # group/ver(1) + buf_cfg+b2_tail(1) + block_c(2) + block_d(2)
    MEC_ODA_SET: 7,  # group/ver(1) + buf_cfg(1) + aid(2) + msg(2) + timeout(1)
}

# MECs that are variable-length (have an explicit MEL_len byte)
_VARIABLE_MEL: frozenset[int] = frozenset({MEC_RT, MEC_AF})

# Global MECs: no DSN/PSN fields, variable-length with explicit mel_len byte
_GLOBAL_MECS: frozenset[int] = frozenset({0x0D, 0x19, 0x1C, 0x1E, 0x23, 0x27, 0x2C,
                                           MEC_ODA_DATA})

# MECs that have DSN but NO PSN field on the wire
_NO_PSN_MECS: frozenset[int] = frozenset({MEC_SLC})

# RT status-byte field masks
RT_BUF_CFG_MASK:    int = 0x60
RT_BUF_CFG_SHIFT:   int = 5
RT_BUF_CLEAR_WRITE: int = 0b00   # clear buffer, write this message
RT_BUF_APPEND:      int = 0b10   # append to buffer (real-world confirmed)
RT_BUF_CLEAR:       int = 0b11   # clear buffer only, text ignored

RT_AB_FLAG_MASK:  int = 0x01
RT_AB_FLAG_SHIFT: int = 0

# ---------------------------------------------------------------------------
# RDS character set (IEC 62106-4:2018 Table 5)
# Maps raw RDS byte codes to Unicode code points.
# Used to decode UECP text fields without loss of special characters.
# ---------------------------------------------------------------------------

_RDS_TO_UNICODE: dict[int, int] = {
    0x00: 0x0000, 0x01: 0x0001, 0x02: 0x0002, 0x03: 0x0003,
    0x04: 0x0004, 0x05: 0x0005, 0x06: 0x0006, 0x07: 0x0007,
    0x08: 0x0008, 0x09: 0x0009, 0x0A: 0x000A, 0x0B: 0x000B,
    0x0C: 0x000C, 0x0D: 0x000D, 0x0E: 0x000E, 0x0F: 0x000F,
    0x10: 0x0010, 0x11: 0x0011, 0x12: 0x0012, 0x13: 0x0013,
    0x14: 0x0014, 0x15: 0x0015, 0x16: 0x0016, 0x17: 0x0017,
    0x18: 0x0018, 0x19: 0x0019, 0x1A: 0x001A, 0x1B: 0x001B,
    0x1C: 0x001C, 0x1D: 0x001D, 0x1E: 0x001E, 0x1F: 0x001F,
    0x20: 0x0020, 0x21: 0x0021, 0x22: 0x0022, 0x23: 0x0023,
    0x24: 0x00A4, 0x25: 0x0025, 0x26: 0x0026, 0x27: 0x0027,
    0x28: 0x0028, 0x29: 0x0029, 0x2A: 0x002A, 0x2B: 0x002B,
    0x2C: 0x002C, 0x2D: 0x002D, 0x2E: 0x002E, 0x2F: 0x002F,
    0x30: 0x0030, 0x31: 0x0031, 0x32: 0x0032, 0x33: 0x0033,
    0x34: 0x0034, 0x35: 0x0035, 0x36: 0x0036, 0x37: 0x0037,
    0x38: 0x0038, 0x39: 0x0039, 0x3A: 0x003A, 0x3B: 0x003B,
    0x3C: 0x003C, 0x3D: 0x003D, 0x3E: 0x003E, 0x3F: 0x003F,
    0x40: 0x0040, 0x41: 0x0041, 0x42: 0x0042, 0x43: 0x0043,
    0x44: 0x0044, 0x45: 0x0045, 0x46: 0x0046, 0x47: 0x0047,
    0x48: 0x0048, 0x49: 0x0049, 0x4A: 0x004A, 0x4B: 0x004B,
    0x4C: 0x004C, 0x4D: 0x004D, 0x4E: 0x004E, 0x4F: 0x004F,
    0x50: 0x0050, 0x51: 0x0051, 0x52: 0x0052, 0x53: 0x0053,
    0x54: 0x0054, 0x55: 0x0055, 0x56: 0x0056, 0x57: 0x0057,
    0x58: 0x0058, 0x59: 0x0059, 0x5A: 0x005A, 0x5B: 0x005B,
    0x5C: 0x005C, 0x5D: 0x005D, 0x5E: 0x2015, 0x5F: 0x005F,
    0x60: 0x2016, 0x61: 0x0061, 0x62: 0x0062, 0x63: 0x0063,
    0x64: 0x0064, 0x65: 0x0065, 0x66: 0x0066, 0x67: 0x0067,
    0x68: 0x0068, 0x69: 0x0069, 0x6A: 0x006A, 0x6B: 0x006B,
    0x6C: 0x006C, 0x6D: 0x006D, 0x6E: 0x006E, 0x6F: 0x006F,
    0x70: 0x0070, 0x71: 0x0071, 0x72: 0x0072, 0x73: 0x0073,
    0x74: 0x0074, 0x75: 0x0075, 0x76: 0x0076, 0x77: 0x0077,
    0x78: 0x0078, 0x79: 0x0079, 0x7A: 0x007A, 0x7B: 0x007B,
    0x7C: 0x007C, 0x7D: 0x007D, 0x7E: 0x203E, 0x7F: 0x007F,
    0x80: 0x00E1, 0x81: 0x00E0, 0x82: 0x00E9, 0x83: 0x00E8,
    0x84: 0x00ED, 0x85: 0x00EC, 0x86: 0x00F3, 0x87: 0x00F2,
    0x88: 0x00FA, 0x89: 0x00F9, 0x8A: 0x00D1, 0x8B: 0x00C7,
    0x8C: 0x015E, 0x8D: 0x00DF, 0x8E: 0x00A1, 0x8F: 0x0132,
    0x90: 0x00E2, 0x91: 0x00E4, 0x92: 0x00EA, 0x93: 0x00EB,
    0x94: 0x00EE, 0x95: 0x00EF, 0x96: 0x00F4, 0x97: 0x00F6,
    0x98: 0x00FB, 0x99: 0x00FC, 0x9A: 0x00F1, 0x9B: 0x00E7,
    0x9C: 0x015F, 0x9D: 0x011F, 0x9E: 0x0131, 0x9F: 0x0133,
    0xA0: 0x00AA, 0xA1: 0x03B1, 0xA2: 0x00A9, 0xA3: 0x2030,
    0xA4: 0x011E, 0xA5: 0x011B, 0xA6: 0x0148, 0xA7: 0x0151,
    0xA8: 0x03C0, 0xA9: 0x20AC, 0xAA: 0x00A3, 0xAB: 0x0024,
    0xAC: 0x2190, 0xAD: 0x2191, 0xAE: 0x2192, 0xAF: 0x2193,
    0xB0: 0x00BA, 0xB1: 0x00B9, 0xB2: 0x00B2, 0xB3: 0x00B3,
    0xB4: 0x00B1, 0xB5: 0x0130, 0xB6: 0x0144, 0xB7: 0x0171,
    0xB8: 0x00B5, 0xB9: 0x00BF, 0xBA: 0x00F7, 0xBB: 0x00B0,
    0xBC: 0x00BC, 0xBD: 0x00BD, 0xBE: 0x00BE, 0xBF: 0x00A7,
    0xC0: 0x00C1, 0xC1: 0x00C0, 0xC2: 0x00C9, 0xC3: 0x00C8,
    0xC4: 0x00CD, 0xC5: 0x00CC, 0xC6: 0x00D3, 0xC7: 0x00D2,
    0xC8: 0x00DA, 0xC9: 0x00D9, 0xCA: 0x0158, 0xCB: 0x010C,
    0xCC: 0x0160, 0xCD: 0x017D, 0xCE: 0x00D0, 0xCF: 0x013F,
    0xD0: 0x00C2, 0xD1: 0x00C4, 0xD2: 0x00CA, 0xD3: 0x00CB,
    0xD4: 0x00CE, 0xD5: 0x00CF, 0xD6: 0x00D4, 0xD7: 0x00D6,
    0xD8: 0x00DB, 0xD9: 0x00DC, 0xDA: 0x0159, 0xDB: 0x010D,
    0xDC: 0x0161, 0xDD: 0x017E, 0xDE: 0x0111, 0xDF: 0x0140,
    0xE0: 0x00C3, 0xE1: 0x00C5, 0xE2: 0x00C6, 0xE3: 0x0152,
    0xE4: 0x0177, 0xE5: 0x00DD, 0xE6: 0x00D5, 0xE7: 0x00D8,
    0xE8: 0x00DE, 0xE9: 0x014A, 0xEA: 0x0154, 0xEB: 0x0106,
    0xEC: 0x015A, 0xED: 0x0179, 0xEE: 0x0166, 0xEF: 0x00F0,
    0xF0: 0x00E3, 0xF1: 0x00E5, 0xF2: 0x00E6, 0xF3: 0x0153,
    0xF4: 0x0175, 0xF5: 0x00FD, 0xF6: 0x00F5, 0xF7: 0x00F8,
    0xF8: 0x00FE, 0xF9: 0x014B, 0xFA: 0x0155, 0xFB: 0x0107,
    0xFC: 0x015B, 0xFD: 0x017A, 0xFE: 0x0167, 0xFF: 0x0000,
}


def rds_bytes_to_str(data: bytes) -> str:
    """Decode raw RDS character-set bytes (IEC 62106-4:2018) to a Unicode string.

    Each byte is mapped through the RDS character table.  Bytes with no
    mapping fall back to a space.  Trailing NUL and space characters are
    stripped (matching common UECP transmitter behaviour).
    """
    chars = []
    for b in data:
        cp = _RDS_TO_UNICODE.get(b)
        if cp is not None and cp != 0:
            chars.append(chr(cp))
        # 0x00 and unmapped bytes are treated as string terminators / stripped
    text = "".join(chars)
    return text.rstrip("\x00 ")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UECPElement:
    """A parsed UECP message element."""
    mec:  int
    dsn:  int   = 0x00
    psn:  int   = 0x00
    data: bytes = b""


@dataclass
class UECPFrame:
    """A parsed, CRC-verified UECP frame."""
    site_addr: int
    enc_addr:  int
    seq:       int
    elements:  list[UECPElement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CRC-CCITT (poly 0x1021, init 0xFFFF)
# ---------------------------------------------------------------------------

def _crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc = ((crc >> 8) | (crc << 8)) & 0xFFFF
        crc ^= byte
        crc ^= ((crc & 0xFF) >> 4)
        crc ^= ((crc << 8) << 4) & 0xFFFF
        crc ^= (((crc & 0xFF) << 4) << 1) & 0xFFFF
    return (crc ^ 0xFFFF) & 0xFFFF


# ---------------------------------------------------------------------------
# Element parsing
# ---------------------------------------------------------------------------

def _parse_elements(mem: bytes) -> list[UECPElement]:
    elements: list[UECPElement] = []
    pos = 0

    while pos < len(mem):
        mec = mem[pos]
        pos += 1

        is_global = mec in _GLOBAL_MECS

        if mec in _NO_ADDR_FIXED_MEL:
            # No DSN, no PSN, fixed-length data immediately after MEC
            dsn = psn = 0x00
            data_len = _NO_ADDR_FIXED_MEL[mec]
            if pos + data_len > len(mem):
                break
            data = mem[pos: pos + data_len]
            pos += data_len
            elements.append(UECPElement(mec=mec, dsn=dsn, psn=psn, data=data))
            continue

        if is_global:
            dsn = psn = 0x00
        elif mec in _NO_PSN_MECS:
            # Has DSN but no PSN field (e.g. SLC 0x1A)
            if pos >= len(mem):
                break
            dsn = mem[pos]
            psn = 0x00
            pos += 1
        else:
            if pos + 2 > len(mem):
                break
            dsn = mem[pos]
            psn = mem[pos + 1]
            pos += 2

        if mec in _FIXED_MEL:
            data_len = _FIXED_MEL[mec]
            if pos + data_len > len(mem):
                break
            data = mem[pos: pos + data_len]
            pos += data_len

        elif mec in _VARIABLE_MEL or is_global:
            if pos >= len(mem):
                break
            mel_len = mem[pos]
            pos += 1
            if pos + mel_len > len(mem):
                break
            data = mem[pos: pos + mel_len]
            pos += mel_len
            # Some senders exclude the status byte from mel_len for RT.
            # When this is the last element, absorb the stray byte.
            if mec == MEC_RT and mel_len > 0 and pos == len(mem) - 1:
                data = data + bytes([mem[pos]])
                pos += 1

        else:
            # Unknown MEC with unknown length — stop parsing
            break

        elements.append(UECPElement(mec=mec, dsn=dsn, psn=psn, data=data))

    return elements


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------

def _parse_frame(frame_body: bytes) -> UECPFrame | None:
    """Parse a de-stuffed frame body.  Returns None if malformed or CRC fails."""
    if len(frame_body) < 6:
        return None

    addr_hi, addr_lo, seq, mel = (frame_body[0], frame_body[1],
                                   frame_body[2], frame_body[3])
    addr = (addr_hi << 8) | addr_lo

    expected_len = 4 + mel + 2
    if len(frame_body) < expected_len:
        return None

    mem = frame_body[4: 4 + mel]
    received_crc = (frame_body[4 + mel] << 8) | frame_body[4 + mel + 1]
    expected_crc = _crc16_ccitt(frame_body[: 4 + mel])
    if received_crc != expected_crc:
        return None

    site_addr = (addr >> 6) & 0x3FF
    enc_addr  = addr & 0x3F
    elements  = _parse_elements(mem)
    return UECPFrame(site_addr=site_addr, enc_addr=enc_addr,
                     seq=seq, elements=elements)


# ---------------------------------------------------------------------------
# Streaming parser
# ---------------------------------------------------------------------------

class UECPParser:
    """Streaming UECP frame parser for a TCP byte stream.

    Feed raw bytes via :meth:`feed`; complete CRC-verified frames are yielded.

    Usage::

        parser = UECPParser()
        for chunk in tcp_stream:
            for frame in parser.feed(chunk):
                handle(frame)
    """

    _SYNC    = 0
    _FRAME   = 1
    _STUFFED = 2

    def __init__(self) -> None:
        self._state = self._SYNC
        self._buf   = bytearray()

    def feed(self, data: bytes) -> Generator[UECPFrame, None, None]:
        for byte in data:
            if self._state == self._SYNC:
                if byte == UECP_STA:
                    self._buf.clear()
                    self._state = self._FRAME

            elif self._state == self._FRAME:
                if byte == UECP_STP:
                    frame = _parse_frame(bytes(self._buf))
                    if frame is not None:
                        yield frame
                    self._buf.clear()
                    self._state = self._SYNC
                elif byte == UECP_STA:
                    self._buf.clear()   # re-sync on unexpected STA
                elif byte == UECP_ESC:
                    self._state = self._STUFFED
                else:
                    self._buf.append(byte)

            elif self._state == self._STUFFED:
                decoded = _STUFFED.get(byte)
                if decoded is not None:
                    self._buf.append(decoded)
                self._state = self._FRAME
