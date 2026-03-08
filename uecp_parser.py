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
