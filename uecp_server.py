"""UECP TCP server for the RDS Master web encoder.

Listens for UECP 6.02 connections and translates incoming messages into
live updates on the shared ``state`` dict.

Supported MECs (groups 0A, 1A, 2A):
    0x01  PI     – Programme Identification
    0x02  PS     – Programme Service name
    0x03  TA/TP  – Traffic Announcement / Traffic Programme
    0x04  DI     – Decoder Identification (stereo, a-head, compressed, dyn-PTY)
    0x05  M/S    – Music/Speech switch
    0x06  PIN    – Programme Item Number (Group 1A Block D)
    0x07  PTY    – Programme Type (0-31)
    0x0A  RT     – RadioText (Group 2A); clear-and-write and clear-only modes
    0x13  AF     – Alternative Frequencies Method A (Group 0A)
    0x1A  SLC    – Slow Labelling Codes (Group 1A Block C; variant 0=ECC, variant 3=LIC)

Text encoding note
------------------
UECP delivers PS and RT as raw RDS character-set bytes (IEC 62106-4:2018).
For the common case (printable Latin characters) the RDS charset is
identical to Latin-1 (ISO 8859-1) in the 0x20-0x7E range, so we decode as
Latin-1 and strip trailing spaces/nulls.  The encoder's text pipeline then
re-encodes the stored Unicode string to RDS bytes when transmitting groups.
"""
from __future__ import annotations

import logging
import socketserver
import threading

from uecp_parser import (
    UECPParser, UECPFrame, UECPElement,
    MEC_PI, MEC_PS, MEC_TA_TP, MEC_DI_PTYI, MEC_MS, MEC_PIN, MEC_PTY, MEC_RT,
    MEC_AF, MEC_SLC, MEC_NAMES,
    RT_BUF_CFG_MASK, RT_BUF_CFG_SHIFT,
    RT_BUF_CLEAR,
)

log = logging.getLogger(__name__)


def _rds_bytes_to_text(data: bytes) -> str:
    """Decode RDS character-set bytes to a Unicode string via Latin-1."""
    return data.decode("latin-1").rstrip("\x00 ")


class UECPStateHandler:
    """Translates UECP frames into live updates on the encoder state dict.

    Parameters
    ----------
    state:
        The shared ``state`` dict from app.py.
    save_callback:
        Called after each frame is processed to persist state to disk.
        Pass ``None`` to skip persistence (useful for testing).
    """

    def __init__(self, state: dict, save_callback=None) -> None:
        self._state = state
        self._save  = save_callback or (lambda: None)
        self._dispatch: dict[int, object] = {
            MEC_PI:      self._handle_pi,
            MEC_PS:      self._handle_ps,
            MEC_TA_TP:   self._handle_ta_tp,
            MEC_DI_PTYI: self._handle_di,
            MEC_MS:      self._handle_ms,
            MEC_PIN:     self._handle_pin,
            MEC_PTY:     self._handle_pty,
            MEC_RT:      self._handle_rt,
            MEC_AF:      self._handle_af,
            MEC_SLC:     self._handle_slc,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle_frame(self, frame: UECPFrame) -> None:
        """Dispatch all elements in a validated frame, then persist state."""
        log.debug(
            "UECP frame site=%d enc=%d seq=%d (%d element(s))",
            frame.site_addr, frame.enc_addr, frame.seq, len(frame.elements),
        )
        changed = False
        for elem in frame.elements:
            handler = self._dispatch.get(elem.mec)
            if handler is None:
                log.debug("UECP: ignoring unhandled MEC 0x%02X (%s)",
                          elem.mec, MEC_NAMES.get(elem.mec, "?"))
                continue
            try:
                handler(elem)  # type: ignore[operator]
                changed = True
            except Exception:
                log.exception("UECP: error handling MEC 0x%02X", elem.mec)

        if changed:
            try:
                self._save()
            except Exception:
                log.exception("UECP: save_callback raised")

    # ------------------------------------------------------------------
    # MEC handlers
    # ------------------------------------------------------------------

    def _handle_pi(self, elem: UECPElement) -> None:
        """0x01 – Programme Identification (2 bytes, big-endian)."""
        if len(elem.data) < 2:
            log.warning("UECP PI: expected 2 bytes, got %d", len(elem.data))
            return
        pi = (elem.data[0] << 8) | elem.data[1]
        self._state["pi"] = f"{pi:04X}"
        log.info("UECP: PI → 0x%04X", pi)

    def _handle_ps(self, elem: UECPElement) -> None:
        """0x02 – Programme Service name (8 raw RDS bytes)."""
        if not elem.data:
            return
        text = _rds_bytes_to_text(elem.data[:8])
        self._state["ps_dynamic"] = text
        log.info("UECP: PS → %r", text)

    def _handle_ta_tp(self, elem: UECPElement) -> None:
        """0x03 – Traffic Announcement / Traffic Programme.

        Data byte:
          bit 1: TP
          bit 0: TA
        """
        if not elem.data:
            log.warning("UECP TA/TP: empty data")
            return
        b = elem.data[0]
        self._state["tp"] = 1 if (b & 0x02) else 0
        self._state["ta"] = 1 if (b & 0x01) else 0
        log.info("UECP: TP=%d TA=%d", self._state["tp"], self._state["ta"])

    def _handle_di(self, elem: UECPElement) -> None:
        """0x04 – Decoder Identification.

        Data byte bits 3-0:
          bit 0: Stereo
          bit 1: Artificial Head
          bit 2: Compressed
          bit 3: Dynamic PTY (PTYI)
        """
        if not elem.data:
            log.warning("UECP DI: empty data")
            return
        di = elem.data[0] & 0x0F
        self._state["di_stereo"] = 1 if (di & 0x1) else 0
        self._state["di_head"]   = 1 if (di & 0x2) else 0
        self._state["di_comp"]   = 1 if (di & 0x4) else 0
        self._state["di_dyn"]    = 1 if (di & 0x8) else 0
        log.info("UECP: DI → 0x%X (stereo=%d head=%d comp=%d dyn=%d)",
                 di,
                 self._state["di_stereo"], self._state["di_head"],
                 self._state["di_comp"],   self._state["di_dyn"])

    def _handle_ms(self, elem: UECPElement) -> None:
        """0x05 – Music/Speech switch.  Bit 0: 1=Music, 0=Speech."""
        if not elem.data:
            log.warning("UECP M/S: empty data")
            return
        self._state["ms"] = 1 if (elem.data[0] & 0x01) else 0
        log.info("UECP: M/S → %s", "Music" if self._state["ms"] else "Speech")

    def _handle_pty(self, elem: UECPElement) -> None:
        """0x07 – Programme Type (bits 4-0, 0-31)."""
        if not elem.data:
            log.warning("UECP PTY: empty data")
            return
        self._state["pty"] = elem.data[0] & 0x1F
        log.info("UECP: PTY → %d", self._state["pty"])

    def _handle_pin(self, elem: UECPElement) -> None:
        """0x06 – Programme Item Number (2 bytes, big-endian → Group 1A Block D).

        Bits 15-11: day (1-31; 0 = not used)
        Bits 10-6:  hour (0-23)
        Bits 5-0:   minute (0-59)

        Receiving a PIN enables Group 1A transmission (en_pin=1).
        """
        if len(elem.data) < 2:
            log.warning("UECP PIN: expected 2 bytes, got %d", len(elem.data))
            return
        pin = (elem.data[0] << 8) | elem.data[1]
        day    = (pin >> 11) & 0x1F
        hour   = (pin >>  6) & 0x1F
        minute =  pin        & 0x3F
        self._state["pin_day"]    = day
        self._state["pin_hour"]   = hour
        self._state["pin_minute"] = minute
        self._state["en_pin"]     = 1
        log.info("UECP: PIN → day=%d %02d:%02d", day, hour, minute)

    def _handle_rt(self, elem: UECPElement) -> None:
        """0x0A – RadioText (Group 2A).

        Status byte (elem.data[0]):
          bits 6-5: buffer config
                    00 = clear buffer and write this message
                    11 = clear buffer only (text ignored)
                    10 = append (treated as clear+write for simplicity)
          bit 0:    A/B flag toggle (informational; scheduler manages A/B)

        RT text (elem.data[1:]) is decoded from RDS character set via Latin-1.
        The RDS end-of-text marker (0x0D) and trailing spaces are stripped.

        The RT Messages list (rt_messages) takes priority over the plain
        rt_text field in the scheduler.  Any UECP RT update therefore also
        clears rt_messages so that UECP has full control of RadioText.
        """
        # Any RT command overrides the message-list system
        self._state["rt_messages"] = "[]"

        if not elem.data:
            self._state["rt_text"] = ""
            log.info("UECP: RT cleared (empty element)")
            return

        status  = elem.data[0]
        buf_cfg = (status & RT_BUF_CFG_MASK) >> RT_BUF_CFG_SHIFT
        rt_raw  = elem.data[1:]

        if buf_cfg == RT_BUF_CLEAR:
            self._state["rt_text"] = ""
            log.info("UECP: RT cleared (buf_cfg=0b11)")
            return

        # CLEAR_WRITE (0b00), APPEND (0b10), and any other value all update
        # rt_text.  The project has no native RT append buffer, so append is
        # treated as a full replacement.
        text = _rds_bytes_to_text(rt_raw).rstrip("\r").rstrip()
        self._state["rt_text"] = text
        log.info("UECP: RT → %r (%d chars, buf_cfg=0b%02b)",
                 text[:32], len(text), buf_cfg)

    def _handle_af(self, elem: UECPElement) -> None:
        """0x13 – Alternative Frequencies, Method A (Group 0A).

        Data format (UECP 6.02):
          bytes 0-1: start address (big-endian; 0 = full replacement)
          byte 2:    Method A count code: 0xE0 | N  (N = 0–25 AF codes)
          bytes 3…3+N-1: FM AF frequency codes
                     code = round((freq_MHz − 87.5) / 0.1), valid range 1–204

        Converts raw FM codes to MHz strings and stores them in
        state['af_list'] (comma-separated, e.g. "87.6, 88.0").
        A count of N=0 clears the AF list and disables AF transmission.
        """
        if not elem.data:
            self._state["af_list"] = ""
            self._state["en_af"] = 0
            log.info("UECP: AF cleared (empty element)")
            return

        if len(elem.data) < 3:
            log.warning("UECP AF: too short (%d bytes, need addr(2)+count(1))",
                        len(elem.data))
            return

        # bytes 0-1: start address — we always do a full replacement
        addr = (elem.data[0] << 8) | elem.data[1]
        if addr != 0:
            log.debug("UECP AF: non-zero start address %d; treating as full replacement", addr)

        # byte 2: count code (0xE0 | N)
        count_code = elem.data[2]
        n = count_code & 0x1F

        if n == 0:
            self._state["af_list"] = ""
            self._state["en_af"] = 0
            log.info("UECP: AF cleared (count code 0x%02X)", count_code)
            return

        codes = elem.data[3: 3 + n]
        freqs = []
        for code in codes:
            if 1 <= code <= 204:
                freq_mhz = 87.5 + code * 0.1
                freqs.append(f"{freq_mhz:.1f}")
            else:
                log.debug("UECP AF: ignoring out-of-range code 0x%02X", code)

        self._state["af_list"] = ", ".join(freqs)
        self._state["en_af"] = 1
        log.info("UECP: AF → %s (%d frequency/frequencies)", self._state["af_list"], len(freqs))

    def _handle_slc(self, elem: UECPElement) -> None:
        """0x1A – Slow Labelling Codes (Group 1A Block C).

        The 2-byte (big-endian) SLC word maps directly to Group 1A Block C:
          bits 14-12: variant code (0–7)
          bits 11-0:  variant-specific data (lower byte used for ECC/LIC)

        Variants handled:
          0  ECC  – Extended Country Code → state['ecc'] (hex string)
          3  LIC  – Language Code         → state['lic'] (hex string)

        Other variants are logged and ignored (the app only transmits
        ECC and LIC in Group 1A).  Receiving any SLC enables Group 1A
        (en_id=1).
        """
        if len(elem.data) < 2:
            log.warning("UECP SLC: expected 2 bytes, got %d", len(elem.data))
            return
        slc_word = (elem.data[0] << 8) | elem.data[1]
        variant  = (slc_word >> 12) & 0x7
        value    =  slc_word & 0xFF

        if variant == 0:
            self._state["ecc"] = f"{value:02X}"
            log.info("UECP: SLC variant 0 (ECC) → 0x%02X", value)
        elif variant == 3:
            self._state["lic"] = f"{value:02X}"
            log.info("UECP: SLC variant 3 (Language) → 0x%02X", value)
        else:
            log.debug("UECP: SLC variant %d data=0x%03X (not mapped to state field)", variant, slc_word & 0xFFF)

        self._state["en_id"] = 1


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------

class _UECPRequestHandler(socketserver.BaseRequestHandler):
    """Handle a single UECP client connection."""

    server: "_UECPServer"

    def handle(self) -> None:
        peer = self.client_address
        log.info("UECP: client connected from %s:%d", *peer)
        parser = UECPParser()
        try:
            while True:
                chunk = self.request.recv(4096)
                if not chunk:
                    break
                for frame in parser.feed(chunk):
                    names = ", ".join(
                        f"0x{e.mec:02X}({MEC_NAMES.get(e.mec,'?')})"
                        for e in frame.elements
                    ) or "(empty)"
                    log.debug("UECP packet from %s:%d: %s", *peer, names)
                    self.server.uecp_handler.handle_frame(frame)
        except OSError as exc:
            log.debug("UECP: connection from %s:%d closed: %s", *peer, exc)
        finally:
            log.info("UECP: client disconnected from %s:%d", *peer)


class _UECPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads      = True
    uecp_handler: UECPStateHandler


class UECPTCPServer:
    """Listens for UECP connections on a TCP port.

    Parameters
    ----------
    host:
        Bind address (e.g. ``"0.0.0.0"`` for all interfaces).
    port:
        TCP port number.
    handler:
        The :class:`UECPStateHandler` to call for each received frame.
    """

    def __init__(self, host: str, port: int, handler: UECPStateHandler) -> None:
        self._server = _UECPServer((host, port), _UECPRequestHandler)
        self._server.uecp_handler = handler
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="uecp-tcp-server",
            daemon=True,
        )

    @property
    def host(self) -> str:
        return self._server.server_address[0]

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def start(self) -> None:
        """Start the TCP server in a background daemon thread."""
        self._thread.start()
        log.info("UECP TCP server listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Gracefully shut down the TCP server."""
        self._server.shutdown()
        self._server.server_close()
        log.info("UECP TCP server stopped")
