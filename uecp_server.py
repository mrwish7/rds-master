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

import json
import logging
import socketserver
import threading

from uecp_parser import (
    UECPParser, UECPFrame, UECPElement,
    MEC_PI, MEC_PS, MEC_TA_TP, MEC_DI_PTYI, MEC_MS, MEC_PIN, MEC_PTY, MEC_RT,
    MEC_AF, MEC_SLC, MEC_FFG, MEC_ODA_SET, MEC_ODA_DATA, MEC_NAMES,
    RT_BUF_CFG_MASK, RT_BUF_CFG_SHIFT,
    RT_BUF_CLEAR,
    rds_bytes_to_str,
)

# MEC 0x24 Free Format Group — buffer config values (bits 6-5 of byte 2)
FFG_BUF_ONCE  = 0b00  # transmit once then discard
FFG_BUF_ADD   = 0b10  # add to cyclic buffer for this group
FFG_BUF_CLEAR = 0b11  # remove all buffered entries for this group

# State fields that UECP handlers are permitted to modify.
# These are snapshotted when the first client connects and restored when the
# last client disconnects, so the encoder always reverts to its pre-UECP
# ("default") content automatically.
_UECP_MANAGED_FIELDS: frozenset[str] = frozenset({
    'pi', 'ps_dynamic',
    'tp', 'ta', 'ms', 'pty',
    'rt_text', 'rt_messages',
    'af_list', 'en_af',
    'ecc', 'lic', 'en_id',
    'en_pin', 'pin_day', 'pin_hour', 'pin_minute',
    'di_stereo', 'di_head', 'di_comp', 'di_dyn',
    'custom_groups', 'custom_oda_list',
    'eon_services',
})

log = logging.getLogger(__name__)


def _rds_bytes_to_text(data: bytes) -> str:
    """Decode RDS character-set bytes (IEC 62106-4:2018) to a Unicode string."""
    return rds_bytes_to_str(data)


class UECPStateHandler:
    """Translates UECP frames into live updates on the encoder state dict.

    Parameters
    ----------
    state:
        The shared ``state`` dict from app.py.
    save_callback:
        Called after each frame is processed (live GUI notification only —
        do NOT persist to disk here; the pre-UECP config must stay on disk).
        Pass ``None`` to skip (useful for testing).
    restore_callback:
        Called after the last UECP client disconnects and the pre-UECP
        snapshot has been restored into ``state``.  Use this to persist
        the restored state to disk and notify the GUI.
    """

    def __init__(self, state: dict, save_callback=None,
                 restore_callback=None) -> None:
        self._state   = state
        self._save    = save_callback    or (lambda: None)
        self._restore = restore_callback or (lambda: None)
        self._snapshot: dict | None = None
        self._client_count = 0
        self._client_lock  = threading.Lock()
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
            MEC_FFG:     self._handle_ffg,
            MEC_ODA_SET:  self._handle_oda_set,
            MEC_ODA_DATA: self._handle_oda_data,
        }
        self._oda_timers: dict[str, threading.Timer] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Called when a UECP client connects.

        On the first connection, a snapshot of all UECP-managed state fields
        is taken so they can be restored when the last client disconnects.
        """
        with self._client_lock:
            self._client_count += 1
            if self._client_count == 1 and self._snapshot is None:
                self._snapshot = {
                    k: self._state[k]
                    for k in _UECP_MANAGED_FIELDS
                    if k in self._state
                }
                log.info("UECP: first client connected — snapshot taken (%d fields)",
                         len(self._snapshot))

    def disconnect(self) -> None:
        """Called when a UECP client disconnects.

        When the last client disconnects, the pre-UECP snapshot is restored
        into state and the restore callback is invoked (typically to persist
        the original config back to disk and notify the GUI).
        """
        with self._client_lock:
            self._client_count = max(0, self._client_count - 1)
            if self._client_count == 0 and self._snapshot is not None:
                self._state.update(self._snapshot)
                self._snapshot = None
                log.info("UECP: last client disconnected — pre-UECP state restored")
                try:
                    self._restore()
                except Exception:
                    log.exception("UECP: restore_callback raised")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def handle_frame(self, frame: UECPFrame) -> None:
        """Dispatch all elements in a validated frame, then persist state."""
        log.debug(
            "UECP frame site=%d enc=%d seq=%d (%d element(s))",
            frame.site_addr, frame.enc_addr, frame.seq, len(frame.elements),
        )
        uecp_psn = int(self._state.get("uecp_psn", 0) or 0)
        uecp_dsn = int(self._state.get("uecp_dsn", 0) or 0)

        changed = False
        for elem in frame.elements:
            # DSN filtering: if a non-zero DSN filter is configured, skip elements
            # whose DSN is non-zero and doesn't match.  DSN=0 always passes ("current").
            if uecp_dsn != 0 and elem.dsn != 0 and elem.dsn != uecp_dsn:
                log.debug("UECP: skipping MEC 0x%02X DSN=%d (filter DSN=%d)",
                          elem.mec, elem.dsn, uecp_dsn)
                continue

            # PSN routing/filtering: only active when uecp_psn is configured and
            # the element carries a non-zero PSN (psn=0 means "current service").
            if uecp_psn != 0 and elem.psn != 0:
                eon_svc = self._find_eon_by_psn(elem.psn)
                if eon_svc is not None:
                    # Route to EON handler for the matching service
                    try:
                        if self._handle_eon_element(elem, eon_svc):
                            changed = True
                    except Exception:
                        log.exception("UECP: error routing MEC 0x%02X to EON PSN=%d",
                                      elem.mec, elem.psn)
                    continue
                elif elem.psn != uecp_psn:
                    # PSN doesn't match main service or any EON — skip
                    log.debug("UECP: skipping MEC 0x%02X PSN=%d (filter PSN=%d, no EON match)",
                              elem.mec, elem.psn, uecp_psn)
                    continue

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


    # ------------------------------------------------------------------
    # PSN/EON routing helpers
    # ------------------------------------------------------------------

    def _find_eon_by_psn(self, psn: int) -> dict | None:
        """Return the first EON service whose ``uecp_psn`` equals ``psn``, or None."""
        try:
            eon_services = json.loads(self._state.get("eon_services", "[]"))
        except Exception:
            return None
        for svc in eon_services:
            svc_psn = int(svc.get("uecp_psn", 0) or 0)
            if svc_psn != 0 and svc_psn == psn:
                return svc
        return None

    def _handle_eon_element(self, elem: UECPElement, svc: dict) -> bool:
        """Apply a UECP element to the EON service matched by PSN.

        Handles MEC_PI, MEC_PS, and MEC_TA_TP (TP bit only — TA is local).
        Returns True if the EON services list was modified.
        """
        psn = int(svc.get("uecp_psn", 0) or 0)
        try:
            eon_services = json.loads(self._state.get("eon_services", "[]"))
        except Exception:
            return False

        # Locate the entry by uecp_psn so we update the live list, not the stale copy
        target_idx = None
        for i, s in enumerate(eon_services):
            if int(s.get("uecp_psn", 0) or 0) == psn:
                target_idx = i
                break
        if target_idx is None:
            return False

        target = eon_services[target_idx]
        modified = False

        if elem.mec == MEC_PI:
            if len(elem.data) >= 2:
                pi = (elem.data[0] << 8) | elem.data[1]
                target["pi_on"] = f"{pi:04X}"
                log.info("UECP EON (PSN=%d): PI(ON) → 0x%04X", psn, pi)
                modified = True
        elif elem.mec == MEC_PS:
            if elem.data:
                text = _rds_bytes_to_text(elem.data[:8])
                target["ps"] = text
                log.info("UECP EON (PSN=%d): PS(ON) → %r", psn, text)
                modified = True
        elif elem.mec == MEC_TA_TP:
            if elem.data:
                b = elem.data[0]
                target["tp"] = 1 if (b & 0x02) else 0
                log.info("UECP EON (PSN=%d): TP(ON) → %d", psn, target["tp"])
                modified = True
        else:
            log.debug("UECP EON (PSN=%d): MEC 0x%02X not routed to EON", psn, elem.mec)

        if modified:
            self._state["eon_services"] = json.dumps(eon_services)
        return modified

    def _handle_ffg(self, elem: UECPElement) -> None:
        """0x24 – Free Format Group.

        Injects freeform data into any RDS group via the custom_groups buffer.

        Wire layout (no DSN/PSN — 6 data bytes directly after MEC):
          elem.data[0] – bits 4-1: group type (0-15), bit 0: version (0=A, 1=B)
          elem.data[1] – bit 7: always 0, bits 6-5: buffer config, bits 4-0: Block B lower 5 bits
          elem.data[2:4] – Block C (16-bit, big-endian)
          elem.data[4:6] – Block D (16-bit, big-endian)

        Buffer config (elem.data[1] bits 6-5):
          0b00  FFG_BUF_ONCE  – transmit once then remove from buffer
          0b01              – reserved, ignored
          0b10  FFG_BUF_ADD  – add to cyclic buffer for this group
          0b11  FFG_BUF_CLEAR – remove all buffered entries for this group
        """
        if len(elem.data) < 6:
            log.warning("UECP FFG: expected 6 data bytes, got %d", len(elem.data))
            return

        group_type = (elem.data[0] >> 1) & 0x0F
        group_ver  = elem.data[0] & 0x01
        buf_cfg    = (elem.data[1] >> 5) & 0x03
        b2_tail    = elem.data[1] & 0x1F
        block_c    = (elem.data[2] << 8) | elem.data[3]
        block_d    = (elem.data[4] << 8) | elem.data[5]
        group_label = f"{group_type}{'A' if group_ver == 0 else 'B'}"

        try:
            existing = json.loads(self._state.get("custom_groups", "[]"))
        except Exception:
            existing = []

        if buf_cfg == FFG_BUF_CLEAR:
            before = len(existing)
            existing = [g for g in existing
                        if not (g.get("type") == group_type
                                and g.get("version") == group_ver
                                and g.get("uecp_ffg"))]
            self._state["custom_groups"] = json.dumps(existing)
            log.info("UECP FFG: cleared buffer for group %s (%d entry/entries removed)",
                     group_label, before - len(existing))

        elif buf_cfg == FFG_BUF_ONCE or buf_cfg == FFG_BUF_ADD:
            if buf_cfg == FFG_BUF_ONCE:
                # Replace any pending one-shot for this group so a rapid stream
                # doesn't accumulate stale entries
                existing = [g for g in existing
                            if not (g.get("type") == group_type
                                    and g.get("version") == group_ver
                                    and g.get("uecp_ffg")
                                    and g.get("one_shot"))]
            entry = {
                "type":         group_type,
                "version":      group_ver,
                "b2_tail":      f"{b2_tail:02X}",
                "b3":           f"{block_c:04X}",
                "b4":           f"{block_d:04X}",
                "enabled":      True,
                "schedule_freq": 1,
                "uecp_ffg":     True,
            }
            if buf_cfg == FFG_BUF_ONCE:
                entry["one_shot"] = True
            existing.append(entry)
            self._state["custom_groups"] = json.dumps(existing)
            log.info("UECP FFG: %s group %s b2=%02X b3=%04X b4=%04X",
                     "one-shot" if buf_cfg == FFG_BUF_ONCE else "buffered",
                     group_label, b2_tail, block_c, block_d)

        else:
            log.debug("UECP FFG: buf_cfg 0b01 (reserved) ignored for group %s", group_label)

    def _handle_oda_set(self, elem: UECPElement) -> None:
        """0x40 – ODA Set (assign ODA AID to a group, announce via Group 3A).

        Wire layout (no DSN/PSN — 7 data bytes directly after MEC):
          elem.data[0]   – bits 5-1: group number (0-31), bit 0: version (0=A, 1=B)
          elem.data[1:3] – ODA Application ID (16-bit, big-endian)
          elem.data[3]   – buffer config (bits 6-5, same values as FFG; ignored for now)
          elem.data[4:6] – Message bits for Group 3A Block C (usually 0x0000)
          elem.data[6]   – data input timeout in minutes (0 = no timeout)

        Adds or replaces an entry in state['custom_oda_list'].  If timeout > 0,
        a background timer removes the entry after the specified number of minutes.
        """
        if len(elem.data) < 7:
            log.warning("UECP ODA_SET: expected 7 data bytes, got %d", len(elem.data))
            return

        group_num = (elem.data[0] >> 1) & 0x1F
        group_ver = elem.data[0] & 0x01
        aid       = (elem.data[1] << 8) | elem.data[2]
        buf_cfg   = (elem.data[3] >> 5) & 0x03
        msg       = (elem.data[4] << 8) | elem.data[5]
        timeout   = elem.data[6]

        # AGTC = (group_number << 1) | version — matches custom_oda_list 'group_type'
        agtc = (group_num << 1) | group_ver
        aid_str = f"{aid:04X}"
        group_label = f"{group_num}{'A' if group_ver == 0 else 'B'}"

        try:
            existing = json.loads(self._state.get("custom_oda_list", "[]"))
        except Exception:
            existing = []

        # Replace any existing UECP-injected entry with the same AID
        existing = [e for e in existing
                    if not (e.get("uecp_oda") and e.get("aid", "").upper() == aid_str)]

        existing.append({
            "group_type": agtc,
            "aid":        aid_str,
            "msg":        f"{msg:04X}",
            "enabled":    True,
            "uecp_oda":   True,
        })
        self._state["custom_oda_list"] = json.dumps(existing)
        log.info("UECP ODA_SET: AID=0x%04X group=%s buf_cfg=0b%02b msg=0x%04X timeout=%dmin",
                 aid, group_label, buf_cfg, msg, timeout)

        # Cancel any existing timeout timer for this AID, then start a new one
        timer_key = f"oda_{aid_str}"
        old_timer = self._oda_timers.pop(timer_key, None)
        if old_timer is not None:
            old_timer.cancel()

        if timeout > 0:
            def _remove_oda(aid_str=aid_str, timeout=timeout):
                try:
                    current = json.loads(self._state.get("custom_oda_list", "[]"))
                    current = [e for e in current
                               if not (e.get("uecp_oda") and e.get("aid", "").upper() == aid_str)]
                    self._state["custom_oda_list"] = json.dumps(current)
                    log.info("UECP ODA_SET: AID=0x%s removed after %dmin timeout",
                             aid_str, timeout)
                except Exception:
                    log.exception("UECP ODA_SET: error removing AID=0x%s on timeout", aid_str)
                finally:
                    self._oda_timers.pop(timer_key, None)

            t = threading.Timer(timeout * 60.0, _remove_oda)
            t.daemon = True
            t.start()
            self._oda_timers[timer_key] = t

    def _handle_oda_data(self, elem: UECPElement) -> None:
        """0x46 – ODA Data (send group data for a registered ODA AID).

        Wire layout (no DSN/PSN — mel_len byte followed by mel_len data bytes):
          mel_len = 0x08 is the only size handled (regular A-type group message).

          elem.data[0:2] – ODA Application ID (16-bit, big-endian)
          elem.data[2]   – config: bit 7=0, bit 6=short-msg(0), bits 5-4=priority (ignored),
                                   bits 3-2=mode (ignored), bits 1-0=buffer config (as FFG)
          elem.data[3]   – bits 4-0: Block B lower 5 bits (b2_tail)
          elem.data[4:6] – Block C (16-bit, big-endian)
          elem.data[6:8] – Block D (16-bit, big-endian)

        The AID must be registered in state['custom_oda_list'] (either manually or via
        MEC40) to resolve which RDS group the data belongs to.  Buffer config behaviour
        is identical to FFG (0b00=once, 0b10=add, 0b11=clear).
        """
        if len(elem.data) != 8:
            log.debug("UECP ODA_DATA: unsupported mel_len=%d (only 8 handled), skipping",
                      len(elem.data))
            return

        aid     = (elem.data[0] << 8) | elem.data[1]
        config  = elem.data[2]
        buf_cfg = config & 0x03
        b2_tail = elem.data[3] & 0x1F
        block_c = (elem.data[4] << 8) | elem.data[5]
        block_d = (elem.data[6] << 8) | elem.data[7]
        aid_str = f"{aid:04X}"

        # Resolve RDS group from the registered ODA list
        try:
            oda_list = json.loads(self._state.get("custom_oda_list", "[]"))
        except Exception:
            oda_list = []

        matching = [o for o in oda_list
                    if o.get("enabled", True) and o.get("aid", "").upper() == aid_str]
        if not matching:
            log.debug("UECP ODA_DATA: AID=0x%04X not found in ODA list, ignoring", aid)
            return

        agtc      = matching[0].get("group_type", 0)
        group_num = (agtc >> 1) & 0x0F
        group_ver =  agtc & 0x01
        group_label = f"{group_num}{'A' if group_ver == 0 else 'B'}"

        try:
            existing = json.loads(self._state.get("custom_groups", "[]"))
        except Exception:
            existing = []

        if buf_cfg == FFG_BUF_CLEAR:
            before = len(existing)
            existing = [g for g in existing
                        if not (g.get("type") == group_num
                                and g.get("version") == group_ver
                                and g.get("uecp_oda_data"))]
            self._state["custom_groups"] = json.dumps(existing)
            log.info("UECP ODA_DATA: cleared buffer for AID=0x%04X group %s (%d removed)",
                     aid, group_label, before - len(existing))

        elif buf_cfg == FFG_BUF_ONCE or buf_cfg == FFG_BUF_ADD:
            if buf_cfg == FFG_BUF_ONCE:
                # Replace any pending one-shot for this group so a rapid stream
                # doesn't accumulate stale entries
                existing = [g for g in existing
                            if not (g.get("type") == group_num
                                    and g.get("version") == group_ver
                                    and g.get("uecp_oda_data")
                                    and g.get("one_shot"))]
            entry = {
                "type":          group_num,
                "version":       group_ver,
                "b2_tail":       f"{b2_tail:02X}",
                "b3":            f"{block_c:04X}",
                "b4":            f"{block_d:04X}",
                "enabled":       True,
                "schedule_freq": 1,
                "uecp_oda_data": True,
            }
            if buf_cfg == FFG_BUF_ONCE:
                entry["one_shot"] = True
            existing.append(entry)
            self._state["custom_groups"] = json.dumps(existing)
            log.info("UECP ODA_DATA: %s AID=0x%04X → group %s b2=%02X b3=%04X b4=%04X",
                     "one-shot" if buf_cfg == FFG_BUF_ONCE else "buffered",
                     aid, group_label, b2_tail, block_c, block_d)

        else:
            log.debug("UECP ODA_DATA: buf_cfg 0b01 (reserved) ignored for AID=0x%04X", aid)


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------

class UECPWebSocketClient:
    """Subscribes to a WebSocket that publishes base64-encoded UECP frames.

    Each message is expected to be a plain-text base64 string containing
    raw UECP frame bytes (the format produced by ws2tcpUECP.py and similar
    tools).  Frames are decoded and dispatched via the shared
    :class:`UECPStateHandler`.

    Reconnects automatically after *reconnect_delay* seconds if the
    connection is lost.

    Parameters
    ----------
    url:
        WebSocket URL, e.g. ``"ws://192.168.1.10/pacific"``.
    handler:
        Shared :class:`UECPStateHandler` instance (may be shared with a
        :class:`UECPTCPServer` so that client counts are pooled correctly).
    reconnect_delay:
        Seconds to wait before attempting to reconnect after a dropped
        connection.
    """

    def __init__(self, url: str, handler: "UECPStateHandler",
                 reconnect_delay: float = 5.0) -> None:
        self._url = url
        self._handler = handler
        self._reconnect_delay = reconnect_delay
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return self._url

    def start(self) -> None:
        """Start the client in a background daemon thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="uecp-ws-client",
            daemon=True,
        )
        self._thread.start()
        log.info("UECP WS client starting: %s", self._url)

    def stop(self) -> None:
        """Signal the client thread to exit; returns immediately."""
        self._stop_event.set()
        log.info("UECP WS client stopping")

    def _run(self) -> None:
        import base64
        try:
            import websocket as _ws_mod
        except ImportError:
            log.error(
                "UECP WS: 'websocket-client' package is not installed. "
                "Install it with:  pip install websocket-client"
            )
            return

        parser = UECPParser()

        while not self._stop_event.is_set():
            ws = None
            connected = False
            try:
                ws = _ws_mod.WebSocket()
                ws.settimeout(5.0)   # short timeout so stop_event is checked regularly
                log.info("UECP WS: connecting to %s", self._url)
                ws.connect(self._url)
                log.info("UECP WS: connected")
                self._handler.connect()
                connected = True

                while not self._stop_event.is_set():
                    try:
                        msg = ws.recv()
                    except _ws_mod.WebSocketTimeoutException:
                        continue   # timeout — loop back to check stop_event
                    if not msg:
                        log.debug("UECP WS: server closed connection")
                        break
                    try:
                        raw = base64.b64decode(
                            msg.encode("utf-8") if isinstance(msg, str) else msg
                        )
                    except Exception:
                        log.debug("UECP WS: skipping non-base64 message")
                        continue
                    for frame in parser.feed(raw):
                        names = ", ".join(
                            f"0x{e.mec:02X}({MEC_NAMES.get(e.mec, '?')})"
                            for e in frame.elements
                        ) or "(empty)"
                        log.debug("UECP WS packet: %s", names)
                        self._handler.handle_frame(frame)

            except Exception as exc:
                log.info("UECP WS: connection error: %s", exc)
            finally:
                if connected:
                    try:
                        self._handler.disconnect()
                    except Exception:
                        log.exception("UECP WS: disconnect() raised")
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

            if not self._stop_event.is_set():
                log.info("UECP WS: reconnecting in %.0fs", self._reconnect_delay)
                self._stop_event.wait(self._reconnect_delay)

        log.info("UECP WS: client stopped")


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------

class _UECPRequestHandler(socketserver.BaseRequestHandler):
    """Handle a single UECP client connection."""

    server: "_UECPServer"

    def handle(self) -> None:
        peer = self.client_address
        log.info("UECP: client connected from %s:%d", *peer)
        self.server.uecp_handler.connect()
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
            self.server.uecp_handler.disconnect()
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
