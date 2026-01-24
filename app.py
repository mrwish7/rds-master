import numpy as np
import sounddevice as sd
import threading
import time
import re
import sys
import os
import signal
import random
import configparser
import urllib.request
import json
from datetime import datetime, timezone, date
from flask import Flask, render_template_string, request, redirect, url_for, session, jsonify
from flask_socketio import SocketIO
from scipy import signal as dsp_signal
from scipy.fft import fft

# --- SETTINGS ---
# Host API filtering: auto-detect based on OS, can be overridden via RDS_HOSTAPI env var
# On Windows, defaults to MME. On other platforms, no filtering.
# Set RDS_HOSTAPI=none to disable filtering, or RDS_HOSTAPI=CoreAudio/ALSA/etc to filter for specific API
_hostapi_env = os.environ.get("RDS_HOSTAPI", "").strip()
if _hostapi_env.lower() == "none":
    REQUIRE_HOSTAPI = None
elif _hostapi_env:
    REQUIRE_HOSTAPI = _hostapi_env
elif sys.platform == 'win32':
    REQUIRE_HOSTAPI = "MME"
else:
    REQUIRE_HOSTAPI = None  # No filtering on macOS/Linux by default

# --- RDS STANDARDS ---
BITRATE = 1187.5
SAMPLE_RATE = 192000
PILOT_FREQ = 19000
RDS_FREQ = 57000
G_POLY = 0x5B9
OFFSETS = {'A': 0x0FC, 'B': 0x198, 'C': 0x168, 'Cp': 0x1E0, 'D': 0x1B4}

# --- DAB CHANNELS (Group 12A ODA Linkage) ---
# Frequency code: (freq_MHz * 1000) / 16 = decimal value -> encode as 16-bit
DAB_CHANNELS = {
    "5A": 0x2AC5, "5B": 0x2B40, "5C": 0x2BBB, "5D": 0x2C36,
    "6A": 0x2CB1, "6B": 0x2D2C, "6C": 0x2DA7, "6D": 0x2E22,
    "7A": 0x2E9D, "7B": 0x2F18, "7C": 0x2F93, "7D": 0x300E,
    "8A": 0x3089, "8B": 0x3104, "8C": 0x317F, "8D": 0x31FA,
    "9A": 0x3275, "9B": 0x32F0, "9C": 0x336B, "9D": 0x33E6,
    "10A": 0x3461, "10B": 0x34DC, "10C": 0x3557, "10D": 0x35D2, "10N": 0x34EC,
    "11A": 0x364D, "11B": 0x36C8, "11C": 0x3743, "11D": 0x37BE, "11N": 0x36D8,
    "12A": 0x3839, "12B": 0x38B4, "12C": 0x392F, "12D": 0x39AA, "12N": 0x38C4,
    "13A": 0x3A25, "13B": 0x3AA0, "13C": 0x3B1B, "13D": 0x3B96, "13E": 0x3C11, "13F": 0x3C8C,
}


# RDS PTY List (Europe/Rest of World)
PTY_LIST_RDS = ["None", "News", "Current Affairs", "Information", "Sport", "Education", "Drama", "Culture", "Science", "Varied", "Pop Music", "Rock Music", "Easy Music", "Light Classical", "Serious Classical", "Other Music", "Weather", "Finance", "Children's", "Social Affairs", "Religion", "Phone-In", "Travel", "Leisure", "Jazz", "Country", "National Music", "Oldies", "Folk Music", "Documentary", "Alarm Test", "Alarm"]

# RBDS PTY List (North America)
PTY_LIST_RBDS = ["None", "News", "Information", "Sport", "Talk", "Rock", "Classic Rock", "Adult Hits", "Soft Rock", "Top 40", "Country", "Oldies", "Soft", "Nostalgia", "Jazz", "Classical", "R&B", "Soft R&B", "Language", "Religious Music", "Religious Talk", "Personality", "Public", "College", "Unassigned", "Unassigned", "Unassigned", "Unassigned", "Unassigned", "Weather", "Emergency Test", "Emergency"]

# Legacy alias for backwards compatibility
PTY_LIST = PTY_LIST_RDS

# --- RT+ CONTENT TYPES (EN 62106 / IEC 62106) ---
RTPLUS_CONTENT_TYPES = {
    0: ("Dummy", "No content type"),
    1: ("Title", "Item title"),
    2: ("Album", "Album/CD name"),
    3: ("Track", "Track number"),
    4: ("Artist", "Artist name"),
    5: ("Composition", "Composition name"),
    6: ("Movement", "Movement name"),
    7: ("Conductor", "Conductor"),
    8: ("Composer", "Composer"),
    9: ("Band", "Band/Orchestra"),
    10: ("Comment", "Free text comment"),
    11: ("Genre", "Genre"),
    12: ("News", "News headlines"),
    13: ("News.Local", "Local news"),
    14: ("Stock", "Stock market"),
    15: ("Sport", "Sport news"),
    16: ("Lottery", "Lottery numbers"),
    17: ("Horoscope", "Horoscope"),
    18: ("Daily", "Daily diversion"),
    19: ("Health", "Health tips"),
    20: ("Event", "Event info"),
    21: ("Scene", "Scene/Film info"),
    22: ("Cinema", "Cinema info"),
    23: ("TV", "TV info"),
    24: ("DateTime", "Date/Time"),
    25: ("Weather", "Weather info"),
    26: ("Traffic", "Traffic info"),
    27: ("Alarm", "Alarm/Emergency"),
    28: ("Advert", "Advertisement"),
    29: ("URL", "Website URL"),
    30: ("Other", "Other info"),
    31: ("Stn.Short", "Station name short"),
    32: ("Stn.Long", "Station name long"),
    33: ("Prog.Now", "Current program"),
    34: ("Prog.Next", "Next program"),
    35: ("Prog.Part", "Program part"),
    36: ("Host", "Host name"),
    37: ("Editorial", "Editorial staff"),
    38: ("Frequency", "Frequency info"),
    39: ("Homepage", "Homepage URL"),
    40: ("Subchannel", "Sub-channel"),
    41: ("Phone.Hotline", "Hotline phone"),
    42: ("Phone.Studio", "Studio phone"),
    43: ("Phone.Other", "Other phone"),
    44: ("SMS.Studio", "Studio SMS"),
    45: ("SMS.Other", "Other SMS"),
    46: ("Email.Hotline", "Hotline email"),
    47: ("Email.Studio", "Studio email"),
    48: ("Email.Other", "Other email"),
    49: ("MMS.Phone", "MMS number"),
    50: ("Chat", "Chat"),
    51: ("Chat.Centre", "Chat centre"),
    52: ("Vote.Question", "Vote question"),
    53: ("Vote.Centre", "Vote centre"),
    54: ("RFU", "Reserved"),
    55: ("RFU", "Reserved"),
    56: ("RFU", "Reserved"),
    57: ("RFU", "Reserved"),
    58: ("RFU", "Reserved"),
    59: ("Place", "Place/Location"),
    60: ("Appointment", "Appointment"),
    61: ("Identifier", "Identifier"),
    62: ("Purchase", "Purchase info"),
    63: ("GetData", "Get Data"),
}

app = Flask(__name__)
CONFIG_FILE = 'config.ini'

def get_or_create_secret():
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
        if 'SYSTEM' in config and 'secret_key' in config['SYSTEM']:
            return config['SYSTEM']['secret_key']
    secret = os.urandom(24).hex()
    if not config.has_section('SYSTEM'): config.add_section('SYSTEM')
    config['SYSTEM'] = {'secret_key': secret}
    with open(CONFIG_FILE, 'w') as f: config.write(f)
    return secret

app.secret_key = os.environ.get("RDS_SECRET", get_or_create_secret())
# Restrict CORS to same origin only for security (prevents cross-site WebSocket attacks)
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins=[])

auth_config = {"user": "admin", "pass": "pass"}

# --- DEFAULT STATE ---
default_state = {
    "running": False,
    # Audio
    "device_out_idx": 0, "device_in_idx": -1,
    "genlock": False, "passthrough": False,
    "pilot_level": 0.0, "rds_level": 4.5, "genlock_offset": 0.0,
    
    # Basic
    "pi": "2FFF", "pty": 0, "rbds": False, "tp": 1, "ta": 0, "ms": 1,
    "di_stereo": 1, "di_head": 0, "di_comp": 1, "di_dyn": 0,
    "en_af": 1, "af_list": "87.6, 87.7", "af_method": "A", "af_pairs": "[]",
    "en_eon": 0, "eon_services": "[]",
    
    # Text
    "ps_dynamic": "RDS PRO", "ps_centered": False,
    "rt_text": "RDS MASTER PRO", "rt_manual_buffers": False, "rt_cycle_ab": False,
    "rt_a": "RDS MASTER PRO", "rt_b": "Simple & Open Source RDS Encoder",
    "rt_cr": True, "rt_centered": False,
    "rt_mode": "2A", "rt_cycle": True, "rt_cycle_time": 5, "rt_active_buffer": 0,
    "rt_ab_cycle_count": 2,
    "rt_ab_cycle_count": 2,
    
    # RT+
    "rt_plus_format_a": "{artist} - {title}",
    "rt_plus_format_b": "{artist} - {title}",
    "en_rt_plus": False,
    "rt_plus_mode": "format",  # "format" (legacy) or "builder" (new modal)
    "rt_plus_builder_a": "",   # JSON string with builder config
    "rt_plus_builder_b": "",   # JSON string with builder config

    # RT Messages (unified message list - replaces individual RT fields when used)
    "rt_messages": "[]",  # JSON array of message objects
    
    # Expert
    "ecc": "E3", "lic": "09", "tz_offset": 0.0, "en_ct": 1, "en_id": 1,
    "ps_long_32": "RDS MASTER PRO v10.14", "en_lps": 1, "lps_centered": False, "lps_cr": False,
    "ptyn": "PYTHON", "en_ptyn": 1, "ptyn_centered": False,
    "en_dab": 0, "dab_channel": "12B", "dab_eid": "CE15", "dab_mode": 1, "dab_es_flag": 0,
    "dab_sid": "0000", "dab_variant": 0,

    # Settings
    "auto_start": True,
    "rds_freq": 57000,
    
    # Scheduler
    "group_sequence": "0A 0A 2A 0A", 
    "scheduler_auto": True 
}

state = default_state.copy()
resolved_cache = {"ps_dynamic": "", "ps_long_32": "", "rt_text": "", "rt_a": "", "rt_b": "", "ptyn": ""}

# Global Monitor Data
monitor_data = {
    "ps": "OFF AIR",
    "rt": "",
    "lps": "",
    "ptyn": "",
    "af": "",
    "pi": "0000",
    "pty_idx": 10,
    "rt_plus_info": "",
    "heartbeat": 0,
    "pilot_generated": True
}

# --- RT+ PARSER ---
import json

class RTPlusParser:
    @staticmethod
    def parse(text, fmt_str, centered=False, limit=64, builder_state=None):
        """
        Parse RT+ tags from text.

        Supports two modes:
        1. Format string mode (legacy): Uses {artist}/{title} placeholders
        2. Builder mode (new): Uses explicit builder_state with tag positions
        """
        tags = []
        if not text:
            return tags

        # Calculate centering offset
        offset = 0
        if centered and len(text) < limit:
            offset = (limit - len(text)) // 2

        # Builder mode: use explicit positions from builder state
        if builder_state:
            return RTPlusParser._parse_builder_mode(text, builder_state, offset, limit)

        # Legacy format string mode
        if not fmt_str:
            return tags
        return RTPlusParser._parse_format_mode(text, fmt_str, offset, limit)

    @staticmethod
    def _parse_format_mode(text, fmt_str, offset, limit):
        """Parse using format string pattern matching (legacy mode)."""
        tags = []

        pattern = re.escape(fmt_str)
        pattern = pattern.replace(r"\{artist\}", r"(?P<artist>.+)")
        pattern = pattern.replace(r"\{title\}", r"(?P<title>.+)")

        match = re.search(pattern, text)
        if match:
            for name, _ in match.groupdict().items():
                raw_start = match.start(name)
                length = len(match.group(name))

                real_start = raw_start + offset
                c_type = 1 if name == "title" else 4  # 1=Title, 4=Artist

                if real_start < limit and length > 0:
                    if real_start + length > limit:
                        length = limit - real_start
                    tags.append((c_type, real_start, length))

        return tags

    @staticmethod
    def _resolve_dynamic(text):
        r"""Resolve dynamic source patterns (\\r, \\R, \\w) in text."""
        if not text or "\\" not in text:
            return text
        result = parse_text_source(text)
        return result if result is not None else text

    @staticmethod
    def _parse_builder_mode(text, builder_state, offset, limit):
        """Parse using explicit builder state with position calculation."""
        tags = []

        try:
            if isinstance(builder_state, str):
                if not builder_state.strip():
                    return tags
                builder_state = json.loads(builder_state)
        except:
            return tags

        # Resolve dynamic sources in all text fields
        prefix = RTPlusParser._resolve_dynamic(builder_state.get('prefix', ''))
        tag1_type = int(builder_state.get('tag1_type', -1))
        tag1_text = RTPlusParser._resolve_dynamic(builder_state.get('tag1_text', ''))
        middle = RTPlusParser._resolve_dynamic(builder_state.get('middle', ''))
        tag2_type = int(builder_state.get('tag2_type', -1))
        tag2_text = RTPlusParser._resolve_dynamic(builder_state.get('tag2_text', ''))

        # Calculate Tag 1 position based on resolved text
        if tag1_text and tag1_type >= 0:
            tag1_start = len(prefix) + offset
            tag1_len = len(tag1_text)

            if tag1_start < limit and tag1_len > 0:
                if tag1_start + tag1_len > limit:
                    tag1_len = limit - tag1_start
                tags.append((tag1_type, tag1_start, tag1_len))

        # Calculate Tag 2 position based on resolved text
        if tag2_text and tag2_type >= 0:
            tag2_start = len(prefix) + len(tag1_text) + len(middle) + offset
            tag2_len = len(tag2_text)

            if tag2_start < limit and tag2_len > 0:
                if tag2_start + tag2_len > limit:
                    tag2_len = limit - tag2_start
                tags.append((tag2_type, tag2_start, tag2_len))

        return tags

    @staticmethod
    def build_rt_from_builder(builder_state, resolve=True):
        """Construct RadioText string from builder state."""
        try:
            if isinstance(builder_state, str):
                if not builder_state.strip():
                    return ""
                builder_state = json.loads(builder_state)
        except:
            return ""

        # Resolve dynamic sources if requested
        if resolve:
            prefix = RTPlusParser._resolve_dynamic(builder_state.get('prefix', ''))
            tag1_text = RTPlusParser._resolve_dynamic(builder_state.get('tag1_text', ''))
            middle = RTPlusParser._resolve_dynamic(builder_state.get('middle', ''))
            tag2_text = RTPlusParser._resolve_dynamic(builder_state.get('tag2_text', ''))
            suffix = RTPlusParser._resolve_dynamic(builder_state.get('suffix', ''))
        else:
            prefix = builder_state.get('prefix', '')
            tag1_text = builder_state.get('tag1_text', '')
            middle = builder_state.get('middle', '')
            tag2_text = builder_state.get('tag2_text', '')
            suffix = builder_state.get('suffix', '')

        tag2_type = int(builder_state.get('tag2_type', -1))

        msg = prefix + tag1_text
        if tag2_type >= 0 and tag2_text:
            msg += middle + tag2_text
        msg += suffix

        return msg

# --- DEVICE DISCOVERY ---
def get_valid_devices():
    valid_inputs = []
    valid_outputs = []
    try:
        devs = sd.query_devices()
        apis = sd.query_hostapis()
        for i, d in enumerate(devs):
            api_name = apis[d['hostapi']]['name']
            if REQUIRE_HOSTAPI and REQUIRE_HOSTAPI not in api_name: continue
            try:
                if d['max_output_channels'] > 0:
                    sd.check_output_settings(device=i, samplerate=SAMPLE_RATE)
                    valid_outputs.append({'index': i, 'name': f"{d['name']} ({api_name})"})
                if d['max_input_channels'] > 0:
                    sd.check_input_settings(device=i, samplerate=SAMPLE_RATE)
                    valid_inputs.append({'index': i, 'name': f"{d['name']} ({api_name})"})
            except: continue
    except Exception as e: print(f"Device Error: {e}")
    return valid_inputs, valid_outputs

# --- CONFIG ---
def load_config():
    global state
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
        if 'RDS' in config:
            for k in state:
                if k in config['RDS']:
                    val = config['RDS'][k]
                    if isinstance(default_state[k], bool): state[k] = (val == 'True')
                    elif isinstance(default_state[k], int) and not isinstance(default_state[k], bool): state[k] = int(val)
                    elif isinstance(default_state[k], float): state[k] = float(val)
                    else: state[k] = val
        if 'AUTH' in config:
            auth_config['user'] = config['AUTH'].get('user', auth_config['user'])
            auth_config['pass'] = config['AUTH'].get('pass', auth_config['pass'])
    print("Config loaded.")
    migrate_rt_messages()

    # Initialize monitor_data from loaded state to prevent blank display on startup
    global monitor_data
    monitor_data["pi"] = state.get("pi", "0000")
    monitor_data["pty_idx"] = state.get("pty", 0)
    monitor_data["af"] = state.get("af_list", "")
    monitor_data["ps"] = state.get("ps_dynamic", "RDS PRO")
    monitor_data["rt"] = state.get("rt_text", "")

def migrate_rt_messages():
    """Migrate old RT config fields to new unified rt_messages structure."""
    global state
    try:
        existing = json.loads(state.get("rt_messages", "[]"))
        if existing:
            return  # Already has messages, don't migrate
    except:
        pass

    messages = []
    msg_id = 1

    def parse_legacy_text(text, buffer):
        """Parse legacy RT text with timing syntax into message entries."""
        nonlocal msg_id
        entries = []
        if not text:
            return entries

        # Check for timed syntax: "5s:Msg1 / 10s:Msg2"
        if re.match(r"\s*\d+s:", text):
            for part in re.split(r"\s*/\s*", text):
                m = re.match(r"\s*(\d+)s:(.*)", part)
                if m:
                    cycles = int(m.group(1))
                    content = m.group(2).strip()
                    if content:
                        entries.append({
                            "id": f"msg_{msg_id}",
                            "buffer": buffer,
                            "cycles": cycles,
                            "source_type": "manual",
                            "content": content,
                            "split_delimiter": " - ",
                            "rt_plus_enabled": False,
                            "rt_plus_tags": {"tag1_type": 4, "tag2_type": 1},
                            "enabled": True
                        })
                        msg_id += 1
        else:
            # Single message
            entries.append({
                "id": f"msg_{msg_id}",
                "buffer": buffer,
                "cycles": 2,
                "source_type": "manual",
                "content": text.strip(),
                "split_delimiter": " - ",
                "rt_plus_enabled": False,
                "rt_plus_tags": {"tag1_type": 4, "tag2_type": 1},
                "enabled": True
            })
            msg_id += 1
        return entries

    # Migrate based on mode
    if state.get("rt_manual_buffers"):
        # Manual buffer mode: migrate rt_a and rt_b
        if state.get("rt_a"):
            messages.extend(parse_legacy_text(state["rt_a"], "A"))
        if state.get("rt_b"):
            messages.extend(parse_legacy_text(state["rt_b"], "B"))
    else:
        # Auto mode: migrate rt_text
        if state.get("rt_text"):
            messages.extend(parse_legacy_text(state["rt_text"], "AB"))

    if messages:
        state["rt_messages"] = json.dumps(messages)
        print(f"Migrated {len(messages)} RT message(s) from legacy config.")

def save_config():
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
    config['RDS'] = {k: str(v) for k, v in state.items()}
    config['AUTH'] = {'user': auth_config.get('user','admin'), 'pass': auth_config.get('pass','admin')}
    with open(CONFIG_FILE, 'w') as f: config.write(f)

# --- DATASETS ---
DATASETS_FILE = os.path.join(os.path.dirname(__file__), 'datasets.json')
datasets = {}
current_dataset = 1

def load_datasets():
    global datasets, current_dataset
    try:
        if os.path.exists(DATASETS_FILE):
            with open(DATASETS_FILE, 'r') as f:
                data = json.load(f)
                datasets = data.get('datasets', {})
                current_dataset = data.get('current', 1)
        else:
            datasets = {'1': {'name': 'Dataset 1', 'state': dict(state)}}
            current_dataset = 1
    except:
        datasets = {'1': {'name': 'Dataset 1', 'state': dict(state)}}
        current_dataset = 1

def save_datasets():
    try:
        with open(DATASETS_FILE, 'w') as f:
            json.dump({'datasets': datasets, 'current': current_dataset}, f, indent=2)
    except Exception as e:
        print(f"Error saving datasets: {e}")

def switch_dataset(dataset_num):
    global state, current_dataset
    dataset_num = str(dataset_num)
    if dataset_num in datasets:
        datasets[str(current_dataset)]['state'] = dict(state)
        save_datasets()
        state.update(datasets[dataset_num]['state'])
        current_dataset = int(dataset_num)
        save_config()
        return True
    return False

def sig_abort(sig, frame):
    state["running"] = False
    save_config()
    os._exit(0)
signal.signal(signal.SIGINT, sig_abort)

# --- WORKERS ---

# EBU Latin character mapping for encoding conversion
# RDS uses ISO-8859-1 (Latin-1) encoding for extended characters
def convert_to_ebu_latin(text):
    """Convert UTF-8 text to RDS/EBU Latin (ISO-8859-1) encoding."""
    if not text:
        return ""
    try:
        # Encode to Latin-1, which is what RDS uses
        return text.encode('latin-1', errors='replace').decode('latin-1')
    except:
        # Fallback: replace non-Latin-1 characters with ?
        result = []
        for char in text:
            try:
                char.encode('latin-1')
                result.append(char)
            except:
                result.append('?')
        return ''.join(result)

def parse_text_source(text):
    if not text: return ""
    try:
        if "\\" in text:
            # Track if any substitution failed
            failed = False
            def file_repl(m):
                nonlocal failed
                try: 
                    # Use utf-8-sig to automatically strip BOM, then convert special chars
                    with open(m.group(1), 'r', encoding='utf-8-sig') as f: 
                        content = f.read().strip()
                        return convert_to_ebu_latin(content)
                except: 
                    failed = True
                    return ""  # Return empty on error
            def url_repl(m):
                nonlocal failed
                try:
                    with urllib.request.urlopen(m.group(1), timeout=2) as r: 
                        content = r.read().decode('utf-8').strip()
                        return convert_to_ebu_latin(content)
                except: 
                    failed = True
                    return ""  # Return empty on error
            
            def clean_spaces(s):
                # Normalize any embedded newlines to spaces so RT doesn't break lines unexpectedly
                return s.replace('\r', ' ').replace('\n', ' ')

            t = re.sub(r'\\R"([^"]+)"', lambda m: clean_spaces(file_repl(m)).upper(), text)
            t = re.sub(r'\\r"([^"]+)"', lambda m: clean_spaces(file_repl(m)), t)
            t = re.sub(r'\\w"([^"]+)"', lambda m: clean_spaces(url_repl(m)), t)
            
            # If any source failed, return None to keep cached value
            if failed: return None
            return t
        return text
    except: return text

def text_updater_loop():
    while True:
        if state["running"]:
            for k in ["ps_dynamic", "ps_long_32", "rt_text", "rt_a", "rt_b", "ptyn"]:
                if k in state and "\\" in state[k]: 
                    result = parse_text_source(state[k])
                    if result is not None:  # Only update if parse succeeded
                        resolved_cache[k] = result
        time.sleep(4.0)

def monitor_pusher_loop():
    while True:
        monitor_data["heartbeat"] = int(time.time() * 1000)
        if state["running"]:
            monitor_data["af"] = state["af_list"]
            monitor_data["pty_idx"] = state["pty"]
            monitor_data["pi"] = state["pi"]
            # Pilot generation: disabled if pass-through is enabled and an input device is selected, or when genlock is active
            monitor_data["pilot_generated"] = not ((state.get("passthrough") and state.get("device_in_idx") != -1) or state.get("genlock"))
            socketio.emit('monitor', monitor_data)
        else:
             socketio.emit('monitor', {
                 "ps": "OFF AIR", "rt": "Encoder Stopped", 
                 "lps": "", "ptyn": "", "af": "", "pty_idx": 0, "rt_plus_info": "", "pi": "----",
                 "heartbeat": monitor_data["heartbeat"],
                 "pilot_generated": False
             })
        time.sleep(0.2)

threading.Thread(target=text_updater_loop, daemon=True).start()
threading.Thread(target=monitor_pusher_loop, daemon=True).start()

class Sanitize:
    @staticmethod
    def parse_bool(v):
        if isinstance(v, bool): return v
        if isinstance(v, str): return v.lower() in ('true', '1', 'yes', 'on')
        if isinstance(v, int): return v != 0
        return False

    @staticmethod
    def to_state(data):
        global state
        changed = False
        # Fields that should NOT be converted to EBU Latin (JSON data, mode flags, etc.)
        skip_ebu_fields = {'rt_plus_builder_a', 'rt_plus_builder_b', 'rt_plus_mode', 'rt_messages'}
        for k, v in data.items():
            if k in state:
                try:
                    if isinstance(state[k], bool): state[k] = Sanitize.parse_bool(v)
                    elif isinstance(state[k], float): state[k] = float(v)
                    elif isinstance(state[k], int): state[k] = int(v)
                    elif k in skip_ebu_fields:
                        # Store as-is without EBU Latin conversion
                        state[k] = str(v)
                    else:
                        # Enforce EBU Latin for text fields to ensure spec compliance
                        state[k] = convert_to_ebu_latin(str(v))
                    changed = True
                except: pass
        if changed: save_config()

# --- RDS CORE ---
class RDSHelper:
    @staticmethod
    def crc(data, offset):
        reg = int(data) << 10
        for i in range(16):
            if (reg >> (25)) & 1: reg ^= (G_POLY << 15)
            reg = (reg << 1) & 0x3FFFFFF
        return ((reg >> 16) & 0x3FF) ^ offset

    @staticmethod
    def get_group_bits(g_type, ver, b2_tail, b3_val, b4_val):
        try: pi_v = int(state["pi"], 16)
        except: pi_v = 0x0000
        b1 = (pi_v << 10) | RDSHelper.crc(pi_v, OFFSETS['A'])
        b2_v = (int(g_type) << 12) | (int(ver) << 11) | (int(state["tp"]) << 10) | (int(state["pty"]) << 5) | (int(b2_tail) & 0x1F)
        b2 = (b2_v << 10) | RDSHelper.crc(b2_v, OFFSETS['B'])
        b3 = (int(b3_val) << 10) | RDSHelper.crc(b3_val, OFFSETS['Cp'] if ver else OFFSETS['C'])
        b4 = (int(b4_val) << 10) | RDSHelper.crc(b4_val, OFFSETS['D'])
        bits = []
        for b in [b1, b2, b3, b4]:
            for i in range(25, -1, -1): bits.append((b >> i) & 1)
        return bits

class RDSScheduler:
    def __init__(self):
        self.ps_ptr, self.rt_ptr, self.ptyn_ptr, self.lps_ptr, self.af_ptr = 0, 0, 0, 0, 0
        self.start_time = time.time()
        self.ct_min_lock = -1
        self.last_rt_content = ""
        self.last_rt_text_content = ""
        self.rt_ab_flag = 0
        self.rt_ab_cycles = 0  # number of full RT transmissions before toggling in cycle-ab mode
        self.last_rt_buf = 0   # track last A/B buffer used to reset pointer on change
        self.rt_sequence, self.rt_seq_idx = [], 0
        self.rt_seq_start_time = 0
        self.last_ps_content = ""
        self.ps_sequence, self.ps_seq_idx = [], 0
        self.ps_seq_start_time = 0
        self.burst_counter = 0
        self.last_lps_content = ""
        self.lps_sequence, self.lps_seq_idx = [], 0
        self.lps_seq_start_time = 0
        self.last_ptyn_content = ""
        self.ptyn_sequence, self.ptyn_seq_idx = [], 0
        self.ptyn_seq_start_time = 0
        self.schedule_ptr = 0

        self.rt_plus_toggle = 0
        self.rt_plus_tags = []
        self.last_rt_clean = ""

        self.dab_last_sent = time.time()
        self.group_3a_toggle = 0  # Toggle between DAB ODA and RT+ ODA on Group 3A
        self.schedule_gen_counter = 0  # Counter to half 3A frequency

        # New unified RT message system
        self.rt_msg_idx = 0
        self.rt_msg_cycle_count = 0  # Track completed cycles for current message

        # AF Method B transmission cache
        self.af_b_transmissions = []
        self.last_af_pairs = ""
        self.rt_msg_cache = {}  # Cache resolved content per message id
        self.last_rt_messages_sig = ""  # Track changes to message list

    def get_text(self, key):
        val = state.get(key, "")
        return resolved_cache.get(key, val) if "\\" in val else val

    def get_rt_messages(self):
        """Get enabled RT messages from the unified message list."""
        try:
            messages = json.loads(state.get("rt_messages", "[]"))
            return [m for m in messages if m.get("enabled", True)]
        except:
            return []

    def resolve_msg_content(self, msg):
        """Resolve dynamic content for a message (file/URL sources)."""
        content = msg.get("content", "")
        source_type = msg.get("source_type", "manual")
        prefix = msg.get("prefix", "")
        suffix = msg.get("suffix", "")

        resolved = ""
        if source_type == "manual":
            # Check for inline dynamic patterns
            if "\\" in content:
                resolved = parse_text_source(content) or content
            else:
                resolved = content
        elif source_type == "file":
            # File path - read content
            try:
                with open(content, 'r', encoding='utf-8-sig', errors='replace') as f:
                    resolved = f.read().strip()
            except:
                resolved = ""
        elif source_type == "url":
            # URL - fetch content
            try:
                req = urllib.request.Request(content, headers={'User-Agent': 'RDS-Encoder/1.0'})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    resolved = resp.read().decode('utf-8', errors='replace').strip()
            except:
                resolved = ""
        else:
            resolved = content

        # Apply prefix/suffix for file/URL sources
        if source_type in ("file", "url") and resolved:
            resolved = prefix + resolved + suffix

        return resolved

    def get_current_rt_message(self):
        """Get the current RT message based on cycle count and buffer state."""
        messages = self.get_rt_messages()
        if not messages:
            return None, 0, ""

        # Check if message list changed
        msg_sig = state.get("rt_messages", "[]")
        if msg_sig != self.last_rt_messages_sig:
            self.last_rt_messages_sig = msg_sig
            self.rt_msg_idx = 0
            self.rt_msg_cycle_count = 0
            self.rt_msg_cache = {}
            self.rt_ab_flag = 1 - self.rt_ab_flag  # Toggle on change

        # Get current message
        current_msg = messages[self.rt_msg_idx % len(messages)]

        # Determine buffer
        buffer_setting = current_msg.get("buffer", "AB")
        if buffer_setting == "A":
            buf = 0
        elif buffer_setting == "B":
            buf = 1
        else:  # "AB"
            buf = self.rt_ab_flag

        # Resolve content (with caching for performance during message duration)
        msg_id = current_msg.get("id", "")
        if msg_id not in self.rt_msg_cache:
            self.rt_msg_cache[msg_id] = self.resolve_msg_content(current_msg)
        resolved_content = self.rt_msg_cache[msg_id]

        # If content is empty, skip to next message immediately
        if not resolved_content or not resolved_content.strip():
            self.advance_to_next_rt_message()
            # Retry with next message (limit recursion to prevent infinite loop)
            if len(messages) > 1:
                return self.get_current_rt_message()
            else:
                # Only one message and it's empty - return empty
                return current_msg, buf, ""

        return current_msg, buf, resolved_content

    def advance_to_next_rt_message(self, toggle_buffer=True):
        """Advance to the next RT message and reset cycle count."""
        messages = self.get_rt_messages()
        if not messages:
            return

        self.rt_msg_idx = (self.rt_msg_idx + 1) % len(messages)
        self.rt_msg_cycle_count = 0

        # Clear cache for new message to refresh file/URL content
        current_msg = messages[self.rt_msg_idx % len(messages)]
        msg_id = current_msg.get("id", "")
        if msg_id in self.rt_msg_cache:
            del self.rt_msg_cache[msg_id]

        # Toggle buffer when advancing to next message to prevent overwriting
        # Controlled by toggle_buffer parameter - AB messages don't toggle here since they already toggled
        if toggle_buffer:
            self.rt_ab_flag = 1 - self.rt_ab_flag

    def get_rt_plus_tags_for_message(self, msg, resolved_content, limit):
        """Calculate RT+ tags for a message based on its configuration."""
        if not msg.get("rt_plus_enabled"):
            return []

        tags = []
        tag_config = msg.get("rt_plus_tags", {})
        tag1_type = int(tag_config.get("tag1_type", -1))
        tag2_type = int(tag_config.get("tag2_type", -1))

        # Calculate centering offset
        offset = 0
        if state.get("rt_centered") and len(resolved_content) < limit:
            offset = (limit - len(resolved_content)) // 2

        # Check if manual builder mode (has tag1_text field)
        if msg.get("tag1_text"):
            # Manual builder mode: use explicit positions
            prefix = msg.get("prefix", "")
            tag1_text = msg.get("tag1_text", "")
            middle = msg.get("middle", " - ")
            tag2_text = msg.get("tag2_text", "")

            # Tag 1 position
            if tag1_text and tag1_type >= 0:
                tag1_start = offset + len(prefix)
                tag1_len = min(len(tag1_text), limit - tag1_start)
                if tag1_start < limit and tag1_len > 0:
                    tags.append((tag1_type, tag1_start, tag1_len))

            # Tag 2 position
            if tag2_text and tag2_type >= 0:
                tag2_start = offset + len(prefix) + len(tag1_text) + len(middle)
                tag2_len = min(len(tag2_text), limit - tag2_start)
                if tag2_start < limit and tag2_len > 0:
                    tags.append((tag2_type, tag2_start, tag2_len))
        else:
            # Auto mode: split by delimiter (file/URL sources)
            # For file/URL, we need to account for prefix and suffix in the resolved_content
            prefix = msg.get("prefix", "")
            suffix = msg.get("suffix", "")
            delimiter = msg.get("split_delimiter", " - ")

            # Find where the actual content starts (after prefix) and ends (before suffix)
            content_without_prefix = resolved_content[len(prefix):] if prefix and resolved_content.startswith(prefix) else resolved_content
            # Remove suffix from the end if present
            if suffix and content_without_prefix.endswith(suffix):
                content_without_prefix_or_suffix = content_without_prefix[:-len(suffix)]
            else:
                content_without_prefix_or_suffix = content_without_prefix

            if delimiter and delimiter in content_without_prefix_or_suffix:
                parts = content_without_prefix_or_suffix.split(delimiter, 1)
                # Tag positions include the prefix offset
                tag1_start = offset + len(prefix)
                if len(parts) >= 1 and tag1_type >= 0:
                    tag1_len = min(len(parts[0]), limit - tag1_start)
                    if tag1_start < limit and tag1_len > 0:
                        tags.append((tag1_type, tag1_start, tag1_len))
                if len(parts) >= 2 and tag2_type >= 0:
                    tag2_start = offset + len(prefix) + len(parts[0]) + len(delimiter)
                    # Tag2 length excludes the suffix
                    tag2_len = min(len(parts[1]), limit - tag2_start)
                    if tag2_start < limit and tag2_len > 0:
                        tags.append((tag2_type, tag2_start, tag2_len))
            else:
                # No delimiter found - treat content (without prefix/suffix) as tag1
                if tag1_type >= 0 and content_without_prefix_or_suffix:
                    tag1_start = offset + len(prefix)
                    tag1_len = min(len(content_without_prefix_or_suffix), limit - tag1_start)
                    if tag1_start < limit and tag1_len > 0:
                        tags.append((tag1_type, tag1_start, tag1_len))

        return tags

    def freq_code(self, f):
        try: return round((float(f) - 87.5) / 0.1) if 87.6 <= float(f) <= 107.9 else 205
        except: return 205

    def split(self, text, width=8, center=False):
        pad = lambda s: s.center(width) if center else s.ljust(width)
        # PS case (width<=8): preserve exact single short frame; otherwise word-wrap without splitting words
        if width <= 8:
            if text is None: return [pad("")]
            if len(text) <= width:
                return [pad(text)]
            # Word-wrap for readability, avoid splitting words; split overlong words only
            words, frames, curr = text.split(), [], ""
            for w in words:
                if len(w) > width:
                    if curr:
                        frames.append(pad(curr)); curr = ""
                    chunks = [w[i:i+width] for i in range(0, len(w), width)]
                    for c in chunks[:-1]:
                        frames.append(pad(c))
                    curr = chunks[-1]
                else:
                    test = (curr + " " + w).strip() if curr else w
                    if len(test) <= width:
                        curr = test
                    else:
                        # Optional heuristic: if current is very short (<4), keep it alone
                        frames.append(pad(curr)); curr = w
            if curr:
                frames.append(pad(curr))
            return frames

        # Wider fields (e.g., RT): keep existing word-wrap behavior
        words, frames, curr = text.split(), [], ""
        for w in words:
            if len(w) > width:
                if curr:
                    frames.append(pad(curr)); curr = ""
                chunks = [w[i:i+width] for i in range(0, len(w), width)]
                for c in chunks[:-1]:
                    frames.append(pad(c))
                curr = chunks[-1]
            else:
                test = (curr + " " + w).strip() if curr else w
                if len(test) <= width:
                    curr = test
                else:
                    frames.append(pad(curr)); curr = w
        if curr:
            frames.append(pad(curr))
        return frames

    def parse_smart(self, raw, width, center):
        seq = []
        if re.match(r"\s*\d+s:", raw):
            # Only treat timed syntax when the string starts with Ns: tokens.
            # Split on whitespace-delimited slashes so literal slashes remain intact.
            for p in re.split(r"\s*/\s*", raw):
                # Preserve trailing spaces in content; allow leading spaces before the number
                m = re.match(r"\s*(\d+)s:(.*)", p)
                if m:
                    content = m.group(2)
                    # Skip entries with blank/empty content (e.g., from empty file reads)
                    if not content.strip():
                        continue
                    for sf in self.split(content, width, center): seq.append((int(m.group(1)), sf))
                else:
                    if not p.strip():
                        continue
                    for sf in self.split(p.strip(), width, center): seq.append((2.5, sf))
        else:
            if width <= 8:
                # Preserve spaces for PS width
                if raw == "" or raw is None: return [(10, " "*width)]
                if len(raw) <= width: return [(10, self.split(raw, width, center)[0])]
                for sf in self.split(raw, width, center): seq.append((2.5, sf))
            else:
                # Trim for RT/others
                if not raw.strip(): return [(10, " "*width)]
                if len(raw.strip()) <= width: return [(10, self.split(raw, width, center)[0])]
                for sf in self.split(raw.strip(), width, center): seq.append((2.5, sf))
        # If all entries were skipped (all blank), return a fallback to avoid empty sequence
        if not seq:
            return [(10, " "*width)]
        return seq

    def parse_schedule_string(self, seq_str):
        out = []
        tokens = seq_str.upper().replace(',', ' ').split()
        for t in tokens:
            match = re.match(r"(\d+)([AB]?)", t)
            if match:
                grp = int(match.group(1))
                ver = 1 if match.group(2) == 'B' else 0
                out.append((grp, ver))
        return out if out else [(0,0)]

    def generate_auto_schedule(self):
        # 65% 0A, 35% 2A (increased 0A by 5% for better PS reception)
        seq = [(0,0), (0,0), (2,0), (0,0), (2,0), (0,0), (0,0), (0,0), (2,0), (0,0), (2,0), (0,0), (2,0), (0,0), (0,0), (2,0), (0,0), (0,0), (2,0), (0,0)]
        if state["en_lps"]: seq.append((15,0)); seq.append((15,0))  # +10% increase
        # EN 50067 recommends up to 8 consecutive 14A/14B groups, reduced to 6 for better PS scheduling
        if state.get("en_eon"):
            for _ in range(6):  # Send 6 Group 14A (reduced from 8 by ~25% for 5% overall reduction)
                seq.append((14,0))
        if state["en_ptyn"]: seq.append((10,0)); seq.append((10,0))  # +5% increase (was +5%, now +10%)
        if state["en_id"]: seq.append((1,0))
        # Half 3A frequency: only add on even counter cycles
        if state.get("en_dab") and (self.schedule_gen_counter % 2 == 0): seq.append((3,0))
        if state["en_rt_plus"]: 
            if self.schedule_gen_counter % 2 == 0: seq.append((3,0))
            seq.append((11,0)) 
        self.schedule_gen_counter += 1
        return seq

    def next(self):
        global monitor_data
        now = datetime.now(timezone.utc)
        
        if state["en_ct"] and now.second == 0 and now.minute != self.ct_min_lock:
            self.ct_min_lock = now.minute
            mjd = (date.today() - date(1858, 11, 17)).days
            b4 = ((now.hour & 0x0F) << 12) | (now.minute << 6) | ((1 if state["tz_offset"]<0 else 0) << 5) | int(abs(state["tz_offset"])*2)
            return RDSHelper.get_group_bits(4, 0, (mjd>>15)&3, ((mjd&0x7FFF)<<1)|((now.hour>>4)&1), b4)

        if state["scheduler_auto"]:
            schedule = self.generate_auto_schedule()
        else:
            schedule = self.parse_schedule_string(state["group_sequence"])

        if self.burst_counter > 0:
            g_type, g_ver = 0, 0
            self.burst_counter -= 1
        else:
            # Check if we should send DAB Group 12A (every 45 seconds)
            if state.get("en_dab") and (time.time() - self.dab_last_sent) >= 45.0:
                g_type, g_ver = 12, 0
                self.dab_last_sent = time.time()
            else:
                g_type, g_ver = schedule[self.schedule_ptr % len(schedule)]
                self.schedule_ptr += 1

        if g_type == 0:
            raw = self.get_text("ps_dynamic")
            sig = f"{raw}_{state['ps_centered']}"
            if sig != self.last_ps_content:
                self.last_ps_content, self.ps_ptr = sig, 0
                if state["scheduler_auto"]: self.burst_counter = 16 
                self.ps_sequence = self.parse_smart(raw, 8, state['ps_centered'])
                self.ps_seq_idx, self.ps_seq_start_time = 0, time.time()
            
            if not self.ps_sequence: self.ps_sequence = [(10, "RDS_PRO ")]
            dur, txt = self.ps_sequence[self.ps_seq_idx % len(self.ps_sequence)]
            
            txt = (txt or "").ljust(8)[:8]
            monitor_data["ps"] = txt

            if (time.time() - self.ps_seq_start_time) >= dur:
                self.ps_seq_idx += 1
                self.ps_seq_start_time, self.ps_ptr = time.time(), 0
                dur, txt = self.ps_sequence[self.ps_seq_idx % len(self.ps_sequence)]
                if state["scheduler_auto"] and self.ps_sequence[self.ps_seq_idx%len(self.ps_sequence)][1] != txt: 
                    self.burst_counter = 12

            seg = self.ps_ptr % 4
            self.ps_ptr += 1
            tail = (state["ta"]<<4)|(state["ms"]<<3)|([state['di_dyn'],state['di_comp'],state['di_head'],state['di_stereo']][seg]<<2)|seg
            b3 = 0xE0E0
            
            if state["en_af"] and g_ver == 0:
                 # Split by comma or space, accept both separators
                 import re
                 afs = [x.strip() for x in re.split(r'[,\s]+', state["af_list"]) if x.strip()]
                 af_method = state.get("af_method", "A")

                 if afs and af_method == "A":
                     # Method A: List all frequencies with count code
                     if self.af_ptr == 0:
                         b3, self.af_ptr = (224+len(afs))<<8 | self.freq_code(afs[0]), 1
                     else:
                         f1 = self.freq_code(afs[self.af_ptr])
                         f2 = self.freq_code(afs[self.af_ptr+1]) if self.af_ptr+1 < len(afs) else 205
                         b3, self.af_ptr = (f1<<8)|f2, (self.af_ptr+2) if self.af_ptr+2 < len(afs) else 0

                 elif afs and af_method == "B":
                     # Method B: Frequency pairs with tuning frequency indicator
                     # Parse af_pairs JSON structure (json already imported at module level)
                     try:
                         af_pairs_json = state.get("af_pairs", "[]")
                         af_pairs = json.loads(af_pairs_json) if af_pairs_json else []
                     except:
                         af_pairs = []

                     # Fallback to af_list format if no pairs defined
                     if not af_pairs and afs:
                         af_pairs = [{
                             'main': afs[0],
                             'alts': ', '.join(afs[1:]) if len(afs) > 1 else '',
                             'regional': False
                         }]

                     if af_pairs:
                        # Check if pairs changed - rebuild transmission list if so
                        if af_pairs_json != self.last_af_pairs:
                            self.last_af_pairs = af_pairs_json
                            self.af_b_transmissions = []
                            self.af_ptr = 0

                            # Build transmission list - Method B
                            # Count ALL frequencies across all pairs (including main freq repetitions)
                            # Group pairs by main frequency
                            freq_groups = {}
                            for pair in af_pairs:
                                main_freq = pair.get('main', '').strip()
                                alts_str = pair.get('alts', '')
                                is_regional = pair.get('regional', False)
                                if not main_freq:
                                    continue

                                alt_freqs = [f.strip() for f in alts_str.split(',') if f.strip()]
                                if not alt_freqs:
                                    continue

                                if main_freq not in freq_groups:
                                    freq_groups[main_freq] = {'alts': [], 'regional': is_regional}

                                freq_groups[main_freq]['alts'].extend(alt_freqs)
                                if is_regional:
                                    freq_groups[main_freq]['regional'] = True

                            # Transmit each frequency group
                            for main_freq, group_data in freq_groups.items():
                                alt_list = group_data['alts']
                                is_regional = group_data['regional']

                                # Count code: 224 + total number of frequency codes transmitted
                                # = 1 (linking freq in count code) + len(alt_list) * 2 (pairs of tuning+alt)
                                total_freqs = 1 + (len(alt_list) * 2)
                                # Use code 250 (filler) for regional variant, otherwise main freq
                                tuning_indicator = 250 if is_regional else self.freq_code(main_freq)
                                self.af_b_transmissions.append(((224 + total_freqs) << 8) | tuning_indicator)

                                # Then transmit all main/alt pairs
                                for alt_freq in alt_list:
                                    self.af_b_transmissions.append(
                                        (self.freq_code(main_freq) << 8) | self.freq_code(alt_freq)
                                    )

                        # Cycle through transmissions
                        if self.af_b_transmissions:
                            b3 = self.af_b_transmissions[self.af_ptr % len(self.af_b_transmissions)]
                            self.af_ptr = (self.af_ptr + 1) % len(self.af_b_transmissions)
                        else:
                            b3 = 0xE0E0
                     else:
                         b3 = 0xE0E0
                         self.af_ptr = 0
            
            if g_ver == 1: b3 = int(state["pi"], 16)
            return RDSHelper.get_group_bits(0, g_ver, tail, b3, (ord(txt[seg*2])<<8)|ord(txt[seg*2+1]))

        elif g_type == 2:
            limit = 32 if state["rt_mode"] == "2B" else 64
            current_msg = None

            # Check for new unified message system first
            rt_messages = self.get_rt_messages()
            if rt_messages:
                # New unified message system
                current_msg, buf, raw = self.get_current_rt_message()
                if not raw:
                    raw = " " * limit  # Fallback to blank

                # Truncate to limit
                raw = raw[:limit]

            elif state["rt_manual_buffers"]:
                # Legacy manual mode: use traditional cycling
                buf = int((time.time()-self.start_time)/state["rt_cycle_time"])%2 if state["rt_cycle"] else state["rt_active_buffer"]
                raw = self.get_text("rt_a" if buf==0 else "rt_b")
            else:
                # Legacy auto mode: single buffer with sequence support
                raw_input = self.get_text("rt_text")

                # Check if we need to rebuild the sequence
                if not self.rt_sequence or raw_input != self.last_rt_text_content:
                    if "/" in raw_input:
                        self.rt_sequence = self.parse_smart(raw_input, limit, False)
                    else:
                        m = re.match(r"\s*(\d+)s:(.*)", raw_input.strip())
                        if m:
                            duration = int(m.group(1))
                            text = m.group(2).strip()[:limit]
                        else:
                            duration = 10
                            text = raw_input.strip()[:limit]
                        self.rt_sequence = [(duration, text)]

                    self.rt_seq_idx = 0
                    self.rt_seq_start_time = time.time()
                    self.last_rt_text_content = raw_input
                    self.rt_ab_flag = 1 - self.rt_ab_flag
                    self.rt_ab_cycles = 0
                    self.rt_ptr = 0

                dur, txt = self.rt_sequence[self.rt_seq_idx % len(self.rt_sequence)]

                if len(self.rt_sequence) > 1 and time.time() - self.rt_seq_start_time >= dur:
                    self.rt_seq_idx += 1
                    self.rt_seq_start_time = time.time()
                    dur, txt = self.rt_sequence[self.rt_seq_idx % len(self.rt_sequence)]
                    if not state["rt_cycle_ab"]:
                        self.rt_ab_flag = 1 - self.rt_ab_flag

                buf = self.rt_ab_flag
                raw = txt.strip()

            # If A/B buffer changed, restart RT from the beginning
            if buf != self.last_rt_buf:
                self.rt_ptr = 0
                self.last_rt_buf = buf

            sig = f"{raw}_{state['rt_centered']}_{state['rt_cr']}"

            # Calculate RT+ tags
            if current_msg:
                # New message system: use per-message RT+ config
                rt_plus_sig = f"{raw}_{current_msg.get('id')}_{current_msg.get('rt_plus_enabled')}_{current_msg.get('split_delimiter')}"
                if raw != self.last_rt_clean or rt_plus_sig != getattr(self, 'last_rtplus_sig', ''):
                    self.last_rt_clean = raw
                    self.last_rtplus_sig = rt_plus_sig
                    self.rt_plus_toggle = 1 - self.rt_plus_toggle
                    self.rt_plus_tags = self.get_rt_plus_tags_for_message(current_msg, raw, limit)

                    tag_str = []
                    display_clean = (raw + '\r') if state["rt_cr"] else raw.center(limit) if state["rt_centered"] else raw.ljust(limit)
                    for t in self.rt_plus_tags:
                        t_name = RTPLUS_CONTENT_TYPES.get(t[0], ("Unknown", ""))[0]
                        content = display_clean[t[1]:t[1]+t[2]]
                        tag_str.append(f"{t_name}: {content}")
                    monitor_data["rt_plus_info"] = " | ".join(tag_str) if tag_str else "(no tags)"
            else:
                # Legacy RT+ handling
                builder_key = "rt_plus_builder_a" if buf == 0 else "rt_plus_builder_b"
                builder_state = state.get(builder_key, "")
                rt_plus_sig = f"{raw}_{state.get('rt_plus_mode')}_{builder_state}_{state.get('rt_plus_format_a')}_{state.get('rt_plus_format_b')}"

                if raw != self.last_rt_clean or rt_plus_sig != getattr(self, 'last_rtplus_sig', ''):
                    self.last_rt_clean = raw
                    self.last_rtplus_sig = rt_plus_sig
                    self.rt_plus_toggle = 1 - self.rt_plus_toggle

                    if state.get('rt_plus_mode') == 'builder' and builder_state:
                        self.rt_plus_tags = RTPlusParser.parse(raw, None, centered=state['rt_centered'], limit=limit, builder_state=builder_state)
                    else:
                        fmt = state["rt_plus_format_a"] if buf == 0 else state["rt_plus_format_b"]
                        self.rt_plus_tags = RTPlusParser.parse(raw, fmt, centered=state['rt_centered'], limit=limit)

                    tag_str = []
                    display_clean = (raw + '\r') if state["rt_cr"] else raw.center(limit) if state["rt_centered"] else raw.ljust(limit)
                    for t in self.rt_plus_tags:
                        t_name = RTPLUS_CONTENT_TYPES.get(t[0], ("Unknown", ""))[0]
                        content = display_clean[t[1]:t[1]+t[2]]
                        tag_str.append(f"{t_name}: {content}")
                    monitor_data["rt_plus_info"] = " | ".join(tag_str) if tag_str else "(no tags)"

            if sig != self.last_rt_content: self.rt_ptr, self.last_rt_content = 0, sig
            clean = (raw + '\r') if state["rt_cr"] else raw.center(limit) if state["rt_centered"] else raw.ljust(limit)

            monitor_data["rt"] = clean

            v = g_ver
            bpg = 2 if v==1 else 4
            if self.rt_ptr * bpg >= len(clean) or (clean.find('\r') != -1 and self.rt_ptr*bpg > clean.find('\r')):
                # Completed one full RT transmission cycle
                if current_msg:
                    # New message system: cycle-based advancement
                    self.rt_msg_cycle_count += 1
                    cycle_limit = current_msg.get("cycles", 2)
                    if cycle_limit < 1:
                        cycle_limit = 1

                    buffer_setting = current_msg.get("buffer", "AB")
                    if buffer_setting == "AB":
                        # For AB messages: do N cycles on buffer A, then N cycles on buffer B
                        # Check if we've completed N cycles on current buffer
                        if self.rt_msg_cycle_count == cycle_limit:
                            # Completed N cycles on buffer A, switch to buffer B
                            self.rt_ab_flag = 1 - self.rt_ab_flag
                        elif self.rt_msg_cycle_count >= cycle_limit * 2:
                            # Completed N cycles on both A and B, advance to next message
                            # Toggle buffer so next message starts on opposite buffer
                            self.advance_to_next_rt_message(toggle_buffer=True)
                    else:
                        # For A or B only messages: just count cycles
                        if self.rt_msg_cycle_count >= cycle_limit:
                            # Toggle buffer so next message doesn't overwrite
                            self.advance_to_next_rt_message(toggle_buffer=True)
                elif not current_msg and state.get("rt_cycle_ab"):
                    # Legacy cycle_ab handling
                    self.rt_ab_cycles += 1
                    try:
                        cycle_target = int(state.get("rt_ab_cycle_count", 2))
                    except:
                        cycle_target = 2
                    if cycle_target < 1: cycle_target = 1
                    if self.rt_ab_cycles >= cycle_target:
                        self.rt_ab_flag = 1 - self.rt_ab_flag
                        self.rt_ab_cycles = 0
                self.rt_ptr = 0
            pad = clean.ljust(64)

            # Encode to Latin-1 bytes for proper RDS character encoding (fixes extended chars like ë, ö, etc.)
            try:
                pad_bytes = pad.encode('latin-1', errors='replace')
            except:
                pad_bytes = pad.encode('ascii', errors='replace')

            # Ensure it's 64 bytes long
            if len(pad_bytes) < 64:
                pad_bytes = pad_bytes + b' ' * (64 - len(pad_bytes))

            a = self.rt_ptr % 16
            self.rt_ptr += 1
            # Group 2B Fix: Block 3 is PI, Block 4 is 2 chars of RT
            # Use byte values directly instead of ord() to properly handle extended characters
            b3_val = (pad_bytes[a*4]<<8)|pad_bytes[a*4+1] if v==0 else int(state["pi"], 16)
            b4_val = (pad_bytes[a*4+2]<<8)|pad_bytes[a*4+3] if v==0 else (pad_bytes[a*2]<<8)|pad_bytes[a*2+1]
            return RDSHelper.get_group_bits(2, v, (buf<<4)|a, b3_val, b4_val)

        elif g_type == 3:
             # Group 3A: ODA announcement for DAB (AID=0x0093) and/or RT+ (AID=0x4BD7)
             if state.get("en_dab") and state.get("en_rt_plus"):
                 # Both enabled: alternate between DAB and RT+
                 self.group_3a_toggle = 1 - self.group_3a_toggle
                 if self.group_3a_toggle == 0:
                     # Announce DAB ODA on Group 12A
                     # Block 2 Tail: Application Group Type Code = 12A = 11000 binary = 0x18
                     b2_tail = 0x18
                     b3_val = 0x0000  # Message bits (reserved)
                     b4_val = 0x0093  # AID for DAB linkage
                     return RDSHelper.get_group_bits(3, 0, b2_tail, b3_val, b4_val)
                 else:
                     # Announce RT+ ODA on Group 11A
                     if state["en_rt_plus"]:
                         return RDSHelper.get_group_bits(3, 0, 22, 0x0000, 0x4BD7)
             elif state.get("en_dab"):
                 # Only DAB enabled: announce Group 12A ODA
                 b2_tail = 0x18
                 b3_val = 0x0000
                 b4_val = 0x0093
                 return RDSHelper.get_group_bits(3, 0, b2_tail, b3_val, b4_val)
             elif state["en_rt_plus"]:
                 # Only RT+ enabled: announce Group 11A ODA
                 return RDSHelper.get_group_bits(3, 0, 22, 0x0000, 0x4BD7)
        
        elif g_type == 11 and state["en_rt_plus"]:
             # --- CORRECTED RT+ PACKING (37 Bits split across blocks) ---
             # Requires 2 tags. If we have 1, we add a dummy. If we have >2, we cycle? 
             # For simplicity, we stick to the first 2 tags found.
             
             t1_typ, t1_start, t1_len = 0, 0, 0
             t2_typ, t2_start, t2_len = 0, 0, 0
             
             tags_to_send = self.rt_plus_tags[:2]
             
             # SMART SORT: Tag 2 has 5-bit length (max 32). Tag 1 has 6-bit (max 64).
             # If we have 2 tags, put the longer one in Tag 1 slot.
             if len(tags_to_send) == 2:
                 if tags_to_send[1][2] > 31 and tags_to_send[0][2] <= 31:
                     tags_to_send.reverse()
                     
             if len(tags_to_send) > 0:
                 t1_typ, t1_start, t1_len = tags_to_send[0]
                 # Spec: Length marker is len-1
                 if t1_len > 0: t1_len -= 1
             
             if len(tags_to_send) > 1:
                 t2_typ, t2_start, t2_len = tags_to_send[1]
                 if t2_len > 0: t2_len -= 1
             
             # Block 2 Tail (5 bits): Toggle(1) | Running(1) | T1_Type_Hi3(3)
             # Note: Running bit is bit 3 (value 8). Toggle is bit 4 (value 16).
             b2_tail = ((self.rt_plus_toggle & 1) << 4) | 0x08 | ((t1_typ >> 3) & 0x07)
             
             # Block 3 (16 bits): T1_Type_Lo3(3) | T1_Start(6) | T1_Len(6) | T2_Type_Hi1(1)
             b3_val = ((t1_typ & 0x07) << 13) | ((t1_start & 0x3F) << 7) | ((t1_len & 0x3F) << 1) | ((t2_typ >> 5) & 0x01)
             
             # Block 4 (16 bits): T2_Type_Lo5(5) | T2_Start(6) | T2_Len(5)
             b4_val = ((t2_typ & 0x1F) << 11) | ((t2_start & 0x3F) << 5) | (t2_len & 0x1F)
             
             return RDSHelper.get_group_bits(11, 0, b2_tail, b3_val, b4_val)

        elif g_type == 14 and state.get("en_eon"):
            # Group 14A: Enhanced Other Networks (EON) - EN 50067 section 3.2.1.8
            eon_services = []
            eon_services_str = state.get("eon_services", "[]")

            try:
                if isinstance(eon_services_str, str):
                    eon_services = json.loads(eon_services_str)
                elif isinstance(eon_services_str, list):
                    eon_services = eon_services_str
                else:
                    eon_services = []
            except Exception as e:
                eon_services = []

            if not eon_services:
                # No services configured - send empty 14A group with filler codes
                return RDSHelper.get_group_bits(14,0,0,0xE0E0,0xE0E0)

            # Initialize EON state tracking
            if not hasattr(self, 'eon_service_idx'):
                self.eon_service_idx = 0
                self.eon_variant = 0  # 0=PS, 4=AF, 13=PTY+TA
                self.eon_ps_seg = 0   # PS segment (0-3)
                self.eon_af_idx = -1  # AF index (-1=count code, 0+=mapped freqs)

            service = eon_services[self.eon_service_idx % len(eon_services)]

            # Get PI(ON) for block 4
            try:
                pi_on = int(service.get('pi_on', 'C000'), 16) & 0xFFFF
            except:
                pi_on = 0xC000

            # Get TP(ON) for block 2
            tp_on = service.get('tp', 0) & 1

            # Variant 0: PS(ON) - 2 characters per transmission, 4 segments total
            if self.eon_variant == 0:
                ps_text = service.get('ps', 'OTHER   ')[:8].ljust(8)

                # Encode directly to bytes - don't use convert_to_ebu_latin as it returns string
                try:
                    ps_bytes = ps_text.encode('latin-1', errors='replace')[:8]
                    if len(ps_bytes) < 8:
                        ps_bytes = ps_bytes.ljust(8, b' ')
                except:
                    ps_bytes = b'OTHER   '

                # Block 2 tail for variant 0: TP(ON) in bit 4, variant bits 000SS where SS=segment
                b2_tail = (tp_on << 4) | (self.eon_ps_seg & 0x03)

                # Send 2 characters for current segment
                b3_val = (ps_bytes[self.eon_ps_seg*2] << 8) | ps_bytes[self.eon_ps_seg*2 + 1]

                # Advance to next PS segment
                self.eon_ps_seg += 1
                if self.eon_ps_seg >= 4:
                    self.eon_ps_seg = 0
                    self.eon_variant = 4  # Next variant: AF
                    self.eon_af_idx = -1

            # Variant 4: AF(ON) - Method A: Tuning freq (TN) + mapped frequencies (ON)
            elif self.eon_variant == 4:
                # Get main station's tuning frequency from af_list
                import re
                main_afs = [x.strip() for x in re.split(r'[,\s]+', state.get("af_list", "")) if x.strip()]
                tuning_freq_str = main_afs[0] if main_afs else "87.6"
                tuning_freq = self.freq_code(tuning_freq_str)

                # Get other network's frequencies (ALL frequencies in af_list are ON freqs)
                af_list = service.get('af_list', '')
                all_afs = [f.strip() for f in af_list.split(',') if f.strip()]

                # Filter out any frequencies that match the tuning frequency to avoid self-pairing
                afs = [f for f in all_afs if self.freq_code(f) != tuning_freq]

                if not afs:
                    # No AFs (or only duplicate of tuning freq) - send filler and skip
                    b2_tail = (tp_on << 4) | 0x04  # Variant 4
                    b3_val = 0xE0E0
                    self.eon_variant = 13  # Next variant: PTY+TA
                else:
                    if self.eon_af_idx == -1:
                        # First transmission: count code + tuning frequency (TN)
                        b2_tail = (tp_on << 4) | 0x04  # Variant 4
                        b3_val = ((224 + len(afs)) << 8) | tuning_freq
                        self.eon_af_idx = 0  # Start at 0 - all afs are ON frequencies
                    else:
                        # Subsequent transmissions: tuning freq (TN) + mapped freq (ON)
                        if self.eon_af_idx < len(afs):
                            b2_tail = (tp_on << 4) | 0x04  # Variant 4
                            mapped_freq = self.freq_code(afs[self.eon_af_idx])
                            b3_val = (tuning_freq << 8) | mapped_freq
                            self.eon_af_idx += 1

                            # Check if we've sent all AFs
                            if self.eon_af_idx >= len(afs):
                                self.eon_variant = 13  # Next variant: PTY+TA
                        else:
                            # Shouldn't happen, but safety check
                            b2_tail = (tp_on << 4) | 0x04
                            b3_val = 0xE0E0
                            self.eon_variant = 13

            # Variant 13: PTY(ON) + TA
            elif self.eon_variant == 13:
                b2_tail = (tp_on << 4) | 0x0D  # Variant 13
                pty_on = service.get('pty', 0) & 0x1F
                ta_on = service.get('ta', 0) & 1
                b3_val = (pty_on << 11) | (ta_on << 10)

                # After PTY+TA, move to next service or back to PS
                self.eon_service_idx += 1
                if self.eon_service_idx >= len(eon_services):
                    self.eon_service_idx = 0
                self.eon_variant = 0  # Start with PS for next service
                self.eon_ps_seg = 0
                self.eon_af_idx = -1

            else:
                # Invalid variant - reset
                b2_tail = (tp_on << 4)
                b3_val = 0xE0E0
                self.eon_variant = 0
                self.eon_ps_seg = 0

            return RDSHelper.get_group_bits(14, 0, b2_tail, b3_val, pi_on)

        elif g_type == 15 and state["en_lps"]:
            raw = self.get_text("ps_long_32")
            lps_centered = state['lps_centered']
            if raw != self.last_lps_content: self.last_lps_content, self.lps_ptr = raw, 0
            if not self.lps_sequence or raw != self.lps_sequence[0][1].strip() or not hasattr(self, 'last_lps_centered') or self.last_lps_centered != lps_centered:
                self.lps_sequence = self.parse_smart(raw, 32, lps_centered)
                self.last_lps_centered = lps_centered
            dur, txt = self.lps_sequence[self.lps_seq_idx % len(self.lps_sequence)]
            
            monitor_data["lps"] = txt + ('\r' if state['lps_cr'] else '')

            if (time.time() - self.lps_seq_start_time) >= dur:
                self.lps_seq_idx += 1
                self.lps_seq_start_time, self.lps_ptr = time.time(), 0
                dur, txt = self.lps_sequence[self.lps_seq_idx % len(self.lps_sequence)]
            if state['lps_cr']:
                # Strip padding and append CR, no null padding
                txt_stripped = txt.rstrip()
                lps_txt = (txt_stripped + '\r').encode('utf-8')
                # Pad to at least 4 bytes for segment transmission, but stop at CR
                if len(lps_txt) < 4:
                    lps_txt = lps_txt.ljust(4, b'\x00')
            else:
                lps_txt = txt.encode('utf-8').ljust(32)[:32]
            seg = self.lps_ptr % 8
            self.lps_ptr += 1
            # For CR mode, only send up to the actual length; for normal mode send all 8 segments
            if state['lps_cr'] and (seg * 4) >= len(lps_txt):
                # Skip this segment and move to next group
                self.schedule_ptr += 1
                return self.next()
            # Pad segment data if needed
            while len(lps_txt) < (seg + 1) * 4:
                lps_txt += b'\x00'
            return RDSHelper.get_group_bits(15, g_ver, seg, (lps_txt[seg*4]<<8)|lps_txt[seg*4+1], (lps_txt[seg*4+2]<<8)|lps_txt[seg*4+3])

        elif g_type == 10 and state["en_ptyn"]:
            raw = self.get_text("ptyn")
            if raw != self.last_ptyn_content: self.last_ptyn_content, self.ptyn_ptr = raw, 0
            if not self.ptyn_sequence or raw != self.ptyn_sequence[0][1].strip(): 
                self.ptyn_sequence = self.parse_smart(raw, 8, state['ptyn_centered'])
            dur, txt = self.ptyn_sequence[self.ptyn_seq_idx % len(self.ptyn_sequence)]
            
            monitor_data["ptyn"] = txt

            if (time.time() - self.ptyn_seq_start_time) >= dur:
                self.ptyn_seq_idx += 1
                self.ptyn_seq_start_time, self.ptyn_ptr = time.time(), 0
                dur, txt = self.ptyn_sequence[self.ptyn_seq_idx % len(self.ptyn_sequence)]
            txt = txt.ljust(8)
            seg = self.ptyn_ptr % 2
            self.ptyn_ptr += 1
            return RDSHelper.get_group_bits(10, g_ver, seg, (ord(txt[seg*4])<<8)|ord(txt[seg*4+1]), (ord(txt[seg*4+2])<<8)|ord(txt[seg*4+3]))
            
        elif g_type == 1 and state["en_id"]:
            vars = [0, 3] 
            vnt = vars[int(time.time()/2) % 2]
            return RDSHelper.get_group_bits(1, g_ver, 0, (vnt << 12) | (int(state['ecc' if vnt==0 else 'lic'], 16) & 0xFF), 0)

        elif g_type == 12 and state.get("en_dab"):
            # Group 12A: ODA data for DAB linkage (Ensemble table)
            # Per EN 301 700 spec section 5.3.3
            dab_ch = state.get("dab_channel", "12B")
            freq_mhz = {
                "5A": 174.928, "5B": 176.640, "5C": 178.352, "5D": 180.064,
                "6A": 181.936, "6B": 183.648, "6C": 185.360, "6D": 187.072,
                "7A": 188.928, "7B": 190.640, "7C": 192.352, "7D": 194.064,
                "8A": 195.936, "8B": 197.648, "8C": 199.360, "8D": 201.072,
                "9A": 202.928, "9B": 204.640, "9C": 206.352, "9D": 208.064,
                "10A": 209.936, "10B": 211.648, "10C": 213.360, "10D": 215.072, "10N": 210.096,
                "11A": 216.928, "11B": 218.640, "11C": 220.352, "11D": 222.064, "11N": 217.088,
                "12A": 223.936, "12B": 225.648, "12C": 227.360, "12D": 229.072, "12N": 224.096,
                "13A": 230.784, "13B": 232.496, "13C": 234.208, "13D": 235.776, "13E": 237.488, "13F": 239.200,
            }.get(dab_ch, 225.648)
            
            # Frequency field: 18-bit value = freq_kHz / 16
            freq_khz = int(freq_mhz * 1000)
            freq_code = freq_khz // 16  # 18-bit value
            
            # Get configurable DAB parameters
            mode = state.get("dab_mode", 1) & 0x03  # 2-bit mode
            es_flag = state.get("dab_es_flag", 0) & 0x01  # E/S flag: 0=ensemble, 1=service
            
            if es_flag == 0:
                # Ensemble table format (E/S = 0)
                # Block 2 Tail (5 bits): freq upper 2 bits (4:3) + mode (2:1) + E/S flag (0)
                freq_upper_2 = (freq_code >> 16) & 0x03
                b2_tail = (freq_upper_2 << 3) | (mode << 1) | es_flag
                
                # Block 3: frequency bits 15-0
                b3_val = freq_code & 0xFFFF
                
                # Block 4: EId (Ensemble ID)
                eid_str = state.get("dab_eid", "CE15")
                try:
                    eid = int(eid_str, 16) & 0xFFFF
                except:
                    eid = 0xCE15  # Default fallback
                b4_val = eid
            else:
                # Service table format (E/S = 1)
                variant = state.get("dab_variant", 0) & 0x0F  # 4-bit variant code
                
                # Block 2 Tail (5 bits): variant (4:1) + E/S flag (0)
                b2_tail = (variant << 1) | es_flag
                
                # Block 3: Info block (depends on variant)
                if variant == 0:
                    # Variant 0: Ensemble information - EId in info block
                    eid_str = state.get("dab_eid", "CE15")
                    try:
                        b3_val = int(eid_str, 16) & 0xFFFF
                    except:
                        b3_val = 0xCE15
                elif variant == 1:
                    # Variant 1: Linkage information (Rfa, LA, S/H, ILS, LSN)
                    # For now, use default values - could be expanded with more UI controls
                    # Format: Rfa(1) LA(1) S/H(1) ILS(1) LSN(12)
                    b3_val = 0x0000  # Default: all flags off, LSN=0
                else:
                    b3_val = 0x0000  # Other variants not implemented
                
                # Block 4: SId (Service ID)
                sid_str = state.get("dab_sid", "0000")
                try:
                    b4_val = int(sid_str, 16) & 0xFFFF
                except:
                    b4_val = 0x0000  # Default fallback
            
            return RDSHelper.get_group_bits(12, 0, b2_tail, b3_val, b4_val)

        elif g_type == 3 and state.get("en_dab"):
            # Group 3A: ODA (Open Data Application) for DAB cross-reference
            # AID: 0xCD46 (DAB linkage)
            dab_ch = state.get("dab_channel", "12B")
            linkage_code = DAB_CHANNELS.get(dab_ch, DAB_CHANNELS["12B"])
            # Block 2 Tail: ODA application variant (0x08 for DAB linkage)
            b2_tail = 0x08
            # Block 3: AID high byte (0xCD) in upper 8 bits
            b3_val = 0xCD00 | 0x38
            # Block 4: AID low byte (0x46) + linkage frequency data
            b4_val = 0x4600 | (linkage_code & 0xFF)
            return RDSHelper.get_group_bits(3, 0, b2_tail, b3_val, b4_val)
        return RDSHelper.get_group_bits(0,0,0,0xE0E0,0xE0E0)

# --- DSP ENGINE ---
class RDSDSP:
    def __init__(self):
        self.sched = RDSScheduler()
        self.p_rds, self.p_pilot, self.bit_clock, self.last_bit = 0.0, 0.0, 0.0, 0
        self.bit_queue = []
        # Tighten filter cutoff to BITRATE (1.1875 kHz) to better approximate spec shaping
        self.taps = dsp_signal.firwin(301, BITRATE * 1.0, fs=SAMPLE_RATE, window=('kaiser', 8.0))
        self.zi = np.zeros(300)

    def process_frame(self, outdata, frames, indata=None):
        lvl_pilot, lvl_rds = (state["pilot_level"]/100.0), (state["rds_level"]/100.0)
        rds_freq = float(state.get("rds_freq", 57000))
        # Clamp to sensible range
        if rds_freq < 56000: rds_freq = 57000
        if rds_freq > 58000: rds_freq = 57000
        
        # Genlock only valid when RDS is 3x pilot (57kHz)
        use_genlock = state["genlock"] and indata is not None and len(indata) == frames and abs(rds_freq - 3*PILOT_FREQ) < 1e-3
        if use_genlock:
             try:
                 mpx_in = indata[:, 0]
                 target_bin = int((PILOT_FREQ / SAMPLE_RATE) * frames)
                 spectrum = np.fft.fft(mpx_in)
                 phase_19k = np.angle(spectrum[target_bin])
                 self.p_rds = (phase_19k * 3.0) + np.deg2rad(state["genlock_offset"])
             except: pass
             
        while len(self.bit_queue) < frames: self.bit_queue.extend(self.sched.next())
        
        bb = np.zeros(frames)
        for i in range(frames):
            self.bit_clock += BITRATE / SAMPLE_RATE
            if self.bit_clock >= 1.0:
                self.bit_clock -= 1.0
                self.last_bit ^= self.bit_queue.pop(0)
            bb[i] = 1.0 if self.last_bit else -1.0
            if self.bit_clock >= 0.5: bb[i] *= -1.0
            
        shaped, self.zi = dsp_signal.lfilter(self.taps, 1.0, bb, zi=self.zi)
        t = np.arange(frames) / SAMPLE_RATE
        
        rds_sig = shaped * np.sin(2 * np.pi * rds_freq * t + self.p_rds) * lvl_rds
        pilot_sig = 0.0
        if not use_genlock and not (state.get("passthrough") and indata is not None):
             pilot_sig = np.sin(2 * np.pi * PILOT_FREQ * t + self.p_pilot) * lvl_pilot
             
             # Phase Lock: If pilot is active, lock RDS phase to pilot (RDS = 3 * Pilot)
             if lvl_pilot > 0:
                 self.p_rds = (self.p_pilot * 3.0) % (2 * np.pi)
                 
             self.p_pilot = (self.p_pilot + 2 * np.pi * PILOT_FREQ * frames / SAMPLE_RATE) % (2 * np.pi)

        self.p_rds = (self.p_rds + 2 * np.pi * rds_freq * frames / SAMPLE_RATE) % (2 * np.pi)
        mixed = rds_sig + pilot_sig
        
        if indata is not None and state["passthrough"] and indata.shape[1] == 2:
             outdata[:] = indata + np.column_stack((mixed, mixed))
        else:
             outdata[:] = np.column_stack((mixed, mixed))

    def callback_duplex(self, indata, outdata, frames, time, status):
        self.process_frame(outdata, frames, indata)

    def callback_output(self, outdata, frames, time, status):
        self.process_frame(outdata, frames, None)

def run_audio():
    engine = RDSDSP()
    try:
        sd_in = state["device_in_idx"] if state["device_in_idx"] != -1 else None
        sd_out = state["device_out_idx"]
        
        if sd_in is not None:
             with sd.Stream(device=(sd_in, sd_out), samplerate=SAMPLE_RATE, blocksize=2048, channels=2, callback=engine.callback_duplex):
                 while state["running"]: sd.sleep(100)
        else:
             with sd.OutputStream(device=sd_out, samplerate=SAMPLE_RATE, blocksize=2048, channels=2, callback=engine.callback_output):
                 while state["running"]: sd.sleep(100)
    except Exception as e:
        print(f"Audio Error: {e}")
        state["running"] = False

# --- UI ---
@app.route('/')
def index():
    if not session.get('auth'): return redirect(url_for('login'))
    inputs, outputs = get_valid_devices()
    return render_template_string(UI_HTML, inputs=inputs, outputs=outputs, state=state, pty_list_rds=PTY_LIST_RDS, pty_list_rbds=PTY_LIST_RBDS, auth_config=auth_config)

@app.route('/login', methods=['GET','POST'])
def login():
    msg = ""
    if request.method == 'POST':
        u = request.form.get('user', '')
        p = request.form.get('pass', '')
        if u == auth_config.get('user') and p == auth_config.get('pass'):
            session['auth'] = True
            return redirect(url_for('index'))
        else:
            msg = "Invalid credentials"
    return render_template_string(LOGIN_HTML, msg=msg, user=auth_config.get('user',''))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/settings', methods=['POST'])
def update_settings():
    if not session.get('auth'): return {"ok": False, "error": "unauthorized"}, 401
    data = request.get_json(silent=True) or {}
    changed = False

    if 'auto_start' in data:
        state['auto_start'] = bool(data['auto_start'])
        changed = True

    user = data.get('user')
    if isinstance(user, str) and user.strip():
        auth_config['user'] = user.strip()
        changed = True

    password = data.get('password')
    if isinstance(password, str) and password.strip():
        auth_config['pass'] = password.strip()
        changed = True

    if changed: save_config()
    return {"ok": True}

@app.route('/resolve-content', methods=['POST'])
def resolve_content():
    """Resolve content from file path or URL for preview."""
    if not session.get('auth'): return {"ok": False, "error": "unauthorized"}, 401
    data = request.get_json(silent=True) or {}
    source_type = data.get('source_type', 'manual')
    content = data.get('content', '')

    if not content:
        return {"ok": True, "resolved": ""}

    resolved = ""
    try:
        if source_type == 'file':
            with open(content, 'r', encoding='utf-8-sig', errors='replace') as f:
                resolved = f.read().strip()
        elif source_type == 'url':
            req = urllib.request.Request(content, headers={'User-Agent': 'RDS-Encoder/1.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                resolved = resp.read().decode('utf-8', errors='replace').strip()
        elif source_type == 'manual':
            # Check for inline dynamic patterns
            if "\\" in content:
                resolved = parse_text_source(content) or content
            else:
                resolved = content
    except Exception as e:
        return {"ok": False, "error": str(e), "resolved": ""}

    return {"ok": True, "resolved": resolved}

@socketio.on('update')
def handle_update(data): 
    if not session.get('auth'): return
    Sanitize.to_state(data)

@socketio.on('control')
def handle_control(data):
    if not session.get('auth'): return
    if data['action'] == 'start':
        state["device_out_idx"] = int(data["dev_out"])
        state["device_in_idx"] = int(data["dev_in"])
        state["running"] = True
        save_config()
        threading.Thread(target=run_audio, daemon=True).start()
    else:
        state["running"] = False
        save_config()

# Dataset API Routes
@app.route('/datasets', methods=['GET'])
def get_datasets():
    if not session.get('auth'): return jsonify({'error': 'Not authenticated'}), 401
    return jsonify({'datasets': datasets, 'current': current_dataset})

@app.route('/datasets/<int:num>', methods=['PUT'])
def update_dataset(num):
    if not session.get('auth'): return jsonify({'error': 'Not authenticated'}), 401
    data = request.json
    datasets[str(num)] = {'name': data.get('name', f'Dataset {num}'), 'state': data.get('state', dict(state))}
    save_datasets()
    return jsonify({'success': True})

@app.route('/datasets/<int:num>/switch', methods=['POST'])
def switch_to_dataset(num):
    if not session.get('auth'): return jsonify({'error': 'Not authenticated'}), 401
    if switch_dataset(num):
        socketio.emit('dataset_changed', {'current': current_dataset})
        return jsonify({'success': True, 'current': current_dataset})
    return jsonify({'error': 'Dataset not found'}), 404

@app.route('/datasets/<int:num>', methods=['DELETE'])
def delete_dataset(num):
    if not session.get('auth'): return jsonify({'error': 'Not authenticated'}), 401
    if str(num) in datasets and len(datasets) > 1:
        del datasets[str(num)]
        save_datasets()
        return jsonify({'success': True})
    return jsonify({'error': 'Cannot delete last dataset'}), 400

load_config()
load_datasets()

def auto_start_if_enabled():
    if state.get("auto_start") and not state.get("running"):
        # Ensure device indices are set from state
        if "device_out_idx" not in state or state["device_out_idx"] is None:
            print("Auto-start: No output device configured, skipping auto-start")
            return
        state["running"] = True
        print(f"Auto-start: Starting RDS encoder (out={state['device_out_idx']}, in={state.get('device_in_idx', -1)})")
        threading.Thread(target=run_audio, daemon=True).start()

# Delay auto-start slightly to ensure Flask/SocketIO is ready
def delayed_auto_start():
    import time
    time.sleep(0.5)  # Wait for server to initialize
    auto_start_if_enabled()

threading.Thread(target=delayed_auto_start, daemon=True).start()

LOGIN_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RDS Master Pro - Login</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: #0f0f10; color: #e5e7eb; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .glass { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 10px 40px rgba(0,0,0,0.35); }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center px-4">
    <div class="glass rounded-md p-8 w-full max-w-sm">
        <div class="text-center mb-6">
            <div class="text-2xl font-bold text-white">RDS <span class="text-pink-400">MASTER</span> PRO</div>
            <div class="text-xs text-gray-400">v10.14 • Secure Login</div>
        </div>
        {% if msg %}
        <div class="mb-4 text-sm text-red-300 bg-red-900/40 border border-red-700 rounded px-3 py-2">{{ msg }}</div>
        {% endif %}
        <form method="POST" class="space-y-4">
            <div>
                <label class="block text-xs text-gray-400 mb-1">Username</label>
                <input name="user" value="{{ user }}" class="w-full bg-black/60 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-pink-500" autofocus>
            </div>
            <div>
                <label class="block text-xs text-gray-400 mb-1">Password</label>
                <input type="password" name="pass" class="w-full bg-black/60 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-pink-500">
            </div>
            <button type="submit" class="w-full bg-pink-600 hover:bg-pink-500 text-white font-semibold rounded py-2 transition">Sign In</button>
        </form>
        <div class="text-[11px] text-gray-500 mt-4">Credentials are loaded from config.ini [AUTH].</div>
    </div>
</body>
</html>
"""

UI_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RDS Master Pro v10.14</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: #1a1a1a; color: #e0e0e0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 12px; overflow: hidden; }
        .app-container { display: flex; height: 100vh; flex-direction: column; }
        .header { background: #2d1b4e; border-bottom: 2px solid #d946ef; padding: 10px 15px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3); }
        .logo { font-size: 18px; font-weight: bold; color: #fff; letter-spacing: 1px; }
        .logo span { color: #d946ef; font-style: italic; }
        .workspace { display: flex; flex: 1; overflow: hidden; }
        .sidebar { width: 180px; background: #222; border-right: 1px solid #333; display: flex; flex-direction: column; }
        .tab-btn { padding: 12px 15px; cursor: pointer; color: #aaa; border-left: 3px solid transparent; transition: all 0.2s; text-align: left; font-weight: 600; font-size: 11px; text-transform: uppercase; }
        .tab-btn:hover { background: #2a2a2a; color: #fff; }
        .tab-btn.active { background: #2d1b4e; color: #d946ef; border-left-color: #d946ef; }
        .content { flex: 1; padding: 20px; overflow-y: auto; background: #1a1a1a; display: none; }
        .content.active { display: block; animation: fadeIn 0.3s; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .section { background: #252525; border: 1px solid #333; border-radius: 4px; margin-bottom: 15px; overflow: hidden; }
        .section-header { background: #333; color: #d946ef; padding: 6px 10px; font-weight: bold; font-size: 11px; text-transform: uppercase; border-bottom: 1px solid #444; display: flex; align-items: center; gap: 5px; }
        .section-body { padding: 10px; display: grid; gap: 10px; }
        input[type="text"], input[type="number"], select, textarea {
            background: #111; border: 1px solid #444; color: #f0f0f0; padding: 4px 8px; font-size: 12px; width: 100%; border-radius: 2px;
        }
        input:focus, select:focus, textarea:focus { border-color: #d946ef; outline: none; }
        label { color: #999; font-size: 10px; text-transform: uppercase; display: block; margin-bottom: 2px; font-weight: 600; }
        .toggle-checkbox { appearance: none; width: 30px; height: 16px; background: #444; border-radius: 10px; position: relative; cursor: pointer; outline: none; }
        .toggle-checkbox:checked { background: #d946ef; }
        .toggle-checkbox::after { content: ''; position: absolute; top: 2px; left: 2px; width: 12px; height: 12px; background: #fff; border-radius: 50%; transition: 0.2s; }
        .toggle-checkbox:checked::after { left: 16px; }
        .pwr-btn {
            background: #1f2937; border: 1px solid #374151; color: #9ca3af; padding: 5px 15px; border-radius: 4px; font-weight: bold; cursor: pointer; transition: 0.3s;
        }
        .pwr-btn.on { background: #d946ef; color: white; border-color: #c026d3; box-shadow: 0 0 10px rgba(217, 70, 239, 0.4); }
        .slider-container { display: flex; align-items: center; gap: 10px; }
        .slider-val { width: 30px; text-align: right; }
        input[type=range] { flex: 1; accent-color: #d946ef; }
        
        .live-display { background: #000; border: 1px solid #333; color: #0f0; font-family: 'Courier New', monospace; padding: 8px; font-size: 14px; font-weight: bold; border-radius: 2px; letter-spacing: 1px; min-height: 20px; }
        .live-display.lg { font-size: 24px; color: #d946ef; text-align: center; letter-spacing: 4px; background: #110011; border-color: #550055; text-transform: uppercase; }
        .live-display.sub { color: #00ffcc; font-size: 12px; }

        /* RT+ Builder Modal */
        .rtplus-modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.8); z-index: 100; display: flex; align-items: center; justify-content: center; padding: 20px; }
        .rtplus-modal { background: #1a1a1a; border: 1px solid #444; border-radius: 8px; max-width: 700px; width: 100%; max-height: 90vh; overflow: hidden; display: flex; flex-direction: column; }
        .rtplus-modal-content { background: #1a1a1a; border: 1px solid #444; border-radius: 8px; width: 100%; max-height: 90vh; overflow-y: auto; padding: 20px; }
        .rtplus-modal-header { background: #2d1b4e; border-bottom: 2px solid #d946ef; padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; }
        .rtplus-modal-body { padding: 16px; overflow-y: auto; flex: 1; }
        .rtplus-modal-footer { background: #252525; border-top: 1px solid #333; padding: 12px 16px; display: flex; justify-content: space-between; }
        .rtplus-tag-1 { background: rgba(251, 146, 60, 0.25); border-bottom: 2px solid #f97316; color: #fdba74; padding: 0 2px; }
        .rtplus-tag-2 { background: rgba(34, 211, 238, 0.25); border-bottom: 2px solid #06b6d4; color: #67e8f9; padding: 0 2px; }
        .rtplus-preview { font-family: 'Courier New', monospace; font-size: 16px; letter-spacing: 1px; background: #0a0a0a; border: 1px solid #333; padding: 12px; border-radius: 4px; min-height: 40px; }

        /* RT Message Cards */
        .rt-msg-card { background: #252525; border: 1px solid #333; border-radius: 4px; overflow: hidden; cursor: move; transition: all 0.2s; }
        .rt-msg-card.disabled { opacity: 0.5; }
        .rt-msg-card.dragging { opacity: 0.5; transform: scale(0.95); }
        .rt-msg-card.drag-over { border-color: #7c3aed; border-width: 2px; margin-top: 4px; }
        .rt-msg-header { display: flex; align-items: center; gap: 8px; padding: 8px 12px; cursor: pointer; }
        .rt-msg-header:hover { background: #2a2a2a; }
        .rt-msg-buffer { padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: bold; }
        .buffer-a { background: #dc2626; color: white; }
        .buffer-b { background: #2563eb; color: white; }
        .buffer-ab { background: #7c3aed; color: white; }
        .rt-msg-duration { color: #888; font-size: 11px; min-width: 30px; }
        .rt-msg-source { font-size: 12px; }
        .rt-msg-preview { flex: 1; color: #ccc; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-family: monospace; }
        .rt-msg-rtplus { color: #f97316; font-size: 10px; font-weight: bold; }
        .rt-msg-actions { display: flex; gap: 4px; }
        .rt-msg-actions button { background: transparent; border: none; color: #666; cursor: pointer; padding: 2px 6px; font-size: 14px; }
        .rt-msg-actions button:hover { color: #fff; }
    </style>
</head>
<body>
    <div class="app-container">
        <div class="header">
            <div class="logo">RDS <span>MASTER PRO</span> <span class="text-[9px] text-gray-500 ml-2">v10.14</span></div>
            <div class="flex items-center gap-4">
                <div class="text-[10px] text-gray-400 flex items-center gap-2">
                    <span id="heartbeat" class="text-xs">♥</span> 192kHz Ready
                </div>
                <button id="pwrBtn" onclick="togglePower()" class="pwr-btn">OFF AIR</button>
                <a href="/logout" class="text-[11px] text-gray-300 hover:text-white underline">Logout</a>
            </div>
        </div>

        <div class="workspace">
            <div class="sidebar">
                <div class="tab-btn active" onclick="setTab('dashboard')">Dashboard</div>
                <div class="tab-btn" onclick="setTab('basic')">Basic RDS</div>
                <div class="tab-btn" onclick="setTab('expert')">Expert</div>
                <div class="tab-btn" onclick="setTab('audio')">Audio & MPX</div>
                <div class="tab-btn" onclick="setTab('datasets')">Datasets</div>
                <div class="tab-btn" onclick="setTab('settings')">Settings</div>
            </div>

            <div id="dashboard" class="content active">
                <div class="grid grid-cols-2 gap-4">
                    <div class="section col-span-2">
                        <div class="section-header">Live Output Monitor (WebSocket)</div>
                        <div class="section-body">
                             <div>
                                 <label>Program Service (PS)</label>
                                 <div class="live-display lg" id="live_ps">OFF AIR</div>
                             </div>
                             
                             <div class="grid grid-cols-4 gap-2">
                                 <div class="col-span-3">
                                     <label>RadioText (RT)</label>
                                     <div class="live-display sub" id="live_rt"></div>
                                 </div>
                                 <div>
                                     <label>PTY</label>
                                     <div class="live-display sub text-center" id="live_pty"></div>
                                 </div>
                             </div>
                             
                             <div>
                                 <label class="flex justify-between"><span>RT+ Status</span> <span class="text-xs text-gray-400">AID: 4BD7 (Group 11A)</span></label>
                                 <div class="live-display sub text-orange-300" id="live_rt_plus"></div>
                             </div>

                             <div class="grid grid-cols-3 gap-2">
                                 <div>
                                     <label>Live PI</label>
                                     <div class="live-display sub text-center text-yellow-300" id="live_pi"></div>
                                 </div>
                                 <div>
                                     <label>Long PS (Group 15)</label>
                                     <div class="live-display sub" id="live_lps"></div>
                                 </div>
                                 <div>
                                     <label>PTY Name</label>
                                     <div class="live-display sub" id="live_ptyn"></div>
                                 </div>
                             </div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="basic" class="content">
                <div class="section">
                    <div class="section-header">Dynamic PS (8-Char)</div>
                    <div class="section-body">
                         <div class="flex justify-between mb-1">
                             <label>Text Source (Supports \R, \w, 3s:)</label>
                             <div class="flex gap-2 items-center"><label>Centre</label><input type="checkbox" class="toggle-checkbox" id="ps_centered" {% if state.ps_centered %}checked{% endif %} onchange="sync()"></div>
                         </div>
                         <input type="text" id="ps_dynamic" value="{{state.ps_dynamic}}" onchange="sync()">
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-header">Station Identification</div>
                    <div class="section-body grid-cols-2">
                        <div><label>PI Code (Hex)</label><input type="text" id="pi" value="{{state.pi}}" maxlength="4" class="font-mono text-center tracking-widest" onchange="sync()"></div>
                        <div>
                            <div class="flex justify-between items-center mb-1">
                                <label>Program Type (PTY)</label>
                                <label class="flex items-center gap-1 text-xs cursor-pointer">
                                    <input type="checkbox" id="rbds" class="toggle-checkbox" {% if state.rbds %}checked{% endif %} onchange="updatePTYList()">
                                    <span>RBDS</span>
                                </label>
                            </div>
                            <select id="pty" onchange="sync()">
                                {% for p in pty_list_rds %}<option value="{{loop.index0}}" {% if loop.index0 == state.pty %}selected{% endif %} data-rds="{{p}}" data-rbds="{{pty_list_rbds[loop.index0]}}">{{p}}</option>{% endfor %}
                            </select>
                        </div>
                    </div>
                </div>
            
                <div class="section">
                    <div class="section-header flex justify-between items-center">
                        <span>RadioText Messages</span>
                        <button onclick="addRTMessage()" class="px-2 py-1 bg-green-700 hover:bg-green-600 rounded text-xs text-white font-bold">+ Add Message</button>
                    </div>
                    <div class="section-body">
                        <!-- Message List -->
                        <div id="rt_messages_list" class="space-y-2 mb-3">
                            <!-- Messages rendered by JavaScript -->
                        </div>

                        <!-- Empty state -->
                        <div id="rt_messages_empty" class="text-center py-4 text-gray-500 text-sm" style="display:none">
                            No messages configured. Click "+ Add Message" to create one.
                        </div>

                        <!-- Global RT Settings -->
                        <div class="flex justify-between items-center bg-[#111] p-2 rounded mt-3">
                             <div class="flex gap-4 items-end">
                                 <div>
                                     <label>Mode</label>
                                     <select id="rt_mode" onchange="sync()"><option value="2A" {% if state.rt_mode == '2A' %}selected{% endif %}>2A (64)</option><option value="2B" {% if state.rt_mode == '2B' %}selected{% endif %}>2B (32)</option></select>
                                 </div>
                             </div>
                             <div class="flex gap-4">
                                 <div class="flex flex-col items-center"><label>RT+ Enable</label><input type="checkbox" class="toggle-checkbox" id="en_rt_plus" {% if state.en_rt_plus %}checked{% endif %} onchange="sync()"></div>
                                 <div class="flex flex-col items-center"><label>Centre</label><input type="checkbox" class="toggle-checkbox" id="rt_centered" {% if state.rt_centered %}checked{% endif %} onchange="if(this.checked) document.getElementById('rt_cr').checked = false; sync()"></div>
                                 <div class="flex flex-col items-center"><label>Append CR</label><input type="checkbox" class="toggle-checkbox" id="rt_cr" {% if state.rt_cr %}checked{% endif %} onchange="sync()"></div>
                             </div>
                        </div>
                    </div>
                </div>

                <!-- RT Message Edit Modal -->
                <div id="rt_msg_modal" class="rtplus-modal-overlay" style="display: none;">
                    <div class="rtplus-modal" style="max-width: 500px;">
                        <div class="rtplus-modal-header">
                            <div>
                                <div class="text-lg font-bold text-white" id="rt_msg_modal_title">Edit Message</div>
                                <div class="text-xs text-gray-400">Configure RT message content and RT+ tags</div>
                            </div>
                            <button onclick="closeRTMsgModal()" class="text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
                        </div>

                        <div class="rtplus-modal-body space-y-4">
                            <input type="hidden" id="rt_msg_edit_id">

                            <!-- Buffer Selection -->
                            <div>
                                <label class="text-xs text-gray-400 mb-1 block">Buffer</label>
                                <div class="flex gap-2">
                                    <button type="button" id="rt_msg_buf_a" onclick="setMsgBuffer('A')" class="px-4 py-2 rounded font-bold text-sm bg-[#333] text-gray-400 hover:bg-[#444]">A</button>
                                    <button type="button" id="rt_msg_buf_b" onclick="setMsgBuffer('B')" class="px-4 py-2 rounded font-bold text-sm bg-[#333] text-gray-400 hover:bg-[#444]">B</button>
                                    <button type="button" id="rt_msg_buf_ab" onclick="setMsgBuffer('AB')" class="px-4 py-2 rounded font-bold text-sm bg-[#7c3aed] text-white">A+B</button>
                                </div>
                            </div>

                            <!-- Cycles -->
                            <div>
                                <label class="text-xs text-gray-400 mb-1 block">Cycles</label>
                                <input type="number" id="rt_msg_cycles" value="2" min="1" max="50" class="w-24 bg-[#111] border border-[#444] rounded px-2 py-1">
                            </div>

                            <!-- Source Type -->
                            <div>
                                <label class="text-xs text-gray-400 mb-1 block">Source Type</label>
                                <div class="flex gap-4">
                                    <label class="flex items-center gap-2 cursor-pointer">
                                        <input type="radio" name="rt_msg_source" value="manual" checked class="accent-[#d946ef]" onchange="updateSourceUI()">
                                        <span class="text-sm text-gray-300">Manual</span>
                                    </label>
                                    <label class="flex items-center gap-2 cursor-pointer">
                                        <input type="radio" name="rt_msg_source" value="file" class="accent-[#d946ef]" onchange="updateSourceUI()">
                                        <span class="text-sm text-gray-300">File</span>
                                    </label>
                                    <label class="flex items-center gap-2 cursor-pointer">
                                        <input type="radio" name="rt_msg_source" value="url" class="accent-[#d946ef]" onchange="updateSourceUI()">
                                        <span class="text-sm text-gray-300">URL</span>
                                    </label>
                                </div>
                            </div>

                            <!-- Content for File/URL -->
                            <div id="rt_msg_content_wrap" style="display:none">
                                <div class="space-y-2">
                                    <div>
                                        <label class="text-xs text-gray-400 mb-1 block" id="rt_msg_content_label">Content Source</label>
                                        <input type="text" id="rt_msg_content" class="w-full bg-[#111] border border-[#444] rounded px-2 py-1" placeholder="Enter file path or URL" oninput="updateMsgPreview()">
                                        <div class="text-[10px] text-gray-500 mt-1" id="rt_msg_content_hint">Full path to file or URL</div>
                                    </div>
                                    <div class="grid grid-cols-2 gap-2">
                                        <div>
                                            <label class="text-xs text-gray-400 mb-1 block">Prefix (optional)</label>
                                            <input type="text" id="rt_msg_prefix_auto" class="w-full bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" placeholder="Text before content" oninput="updateMsgPreview()">
                                        </div>
                                        <div>
                                            <label class="text-xs text-gray-400 mb-1 block">Suffix (optional)</label>
                                            <input type="text" id="rt_msg_suffix_auto" class="w-full bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" placeholder="Text after content" oninput="updateMsgPreview()">
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Manual Mode: Simple Text (when RT+ disabled) -->
                            <div id="rt_msg_manual_simple" style="display:none">
                                <div>
                                    <label class="text-xs text-gray-400 mb-1 block">RadioText Content</label>
                                    <input type="text" id="rt_msg_simple_text" class="w-full bg-[#111] border border-[#444] rounded px-2 py-1" placeholder="Enter RadioText message (max 64 chars)" maxlength="64" oninput="updateMsgPreview()">
                                    <div class="text-[10px] text-gray-500 mt-1">Character count: <span id="rt_msg_simple_count">0</span>/64</div>
                                </div>
                            </div>

                            <!-- Manual Mode: RT+ Builder (when RT+ enabled) -->
                            <div id="rt_msg_manual_builder" style="display:none" class="space-y-3">
                                <div class="flex gap-2 items-center">
                                    <label class="w-16 text-xs text-gray-500 shrink-0">Prefix:</label>
                                    <input type="text" id="rt_msg_prefix" class="flex-1 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" placeholder="Text before tags" oninput="updateMsgPreview()">
                                </div>

                                <div class="bg-[#1a1a1a] border border-orange-900/50 rounded p-2 space-y-2">
                                    <div class="flex gap-2 items-center">
                                        <label class="w-16 text-xs text-orange-400 font-bold shrink-0">Tag 1:</label>
                                        <select id="rt_msg_tag1_type" class="w-32 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" onchange="updateMsgPreview()"></select>
                                    </div>
                                    <div class="flex gap-2 items-center">
                                        <label class="w-16 text-xs text-gray-500 shrink-0">Content:</label>
                                        <input type="text" id="rt_msg_tag1_text" class="flex-1 bg-[#111] border border-orange-900/50 rounded px-2 py-1 text-sm text-orange-300" placeholder="Tag 1 text" oninput="updateMsgPreview()">
                                    </div>
                                </div>

                                <div class="flex gap-2 items-center">
                                    <label class="w-16 text-xs text-gray-500 shrink-0">Between:</label>
                                    <input type="text" id="rt_msg_middle" class="w-24 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm text-center" value=" - " oninput="updateMsgPreview()">
                                </div>

                                <div class="bg-[#1a1a1a] border border-cyan-900/50 rounded p-2 space-y-2">
                                    <div class="flex gap-2 items-center">
                                        <label class="w-16 text-xs text-cyan-400 font-bold shrink-0">Tag 2:</label>
                                        <select id="rt_msg_tag2_type" class="w-32 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" onchange="updateMsgPreview()"></select>
                                    </div>
                                    <div class="flex gap-2 items-center">
                                        <label class="w-16 text-xs text-gray-500 shrink-0">Content:</label>
                                        <input type="text" id="rt_msg_tag2_text" class="flex-1 bg-[#111] border border-cyan-900/50 rounded px-2 py-1 text-sm text-cyan-300" placeholder="Tag 2 text (optional)" oninput="updateMsgPreview()">
                                    </div>
                                </div>

                                <div class="flex gap-2 items-center">
                                    <label class="w-16 text-xs text-gray-500 shrink-0">Suffix:</label>
                                    <input type="text" id="rt_msg_suffix" class="flex-1 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" placeholder="Text after tags" oninput="updateMsgPreview()">
                                </div>
                            </div>

                            <!-- RT+ Configuration (common for all modes) -->
                            <div class="bg-[#1a1a1a] border border-[#333] rounded p-3 space-y-3">
                                <div class="flex justify-between items-center">
                                    <label class="text-xs text-orange-400 font-bold">RT+ Tags</label>
                                    <label class="flex items-center gap-2 cursor-pointer">
                                        <span class="text-xs text-gray-400">Enable for this message</span>
                                        <input type="checkbox" id="rt_msg_rtplus_enabled" class="toggle-checkbox" onchange="updateRTPlusUI()">
                                    </label>
                                </div>

                                <!-- RT+ Options for File/URL -->
                                <div id="rt_msg_rtplus_options" style="display:none" class="space-y-3">
                                    <div class="flex gap-2 items-center">
                                        <label class="w-20 text-xs text-gray-500 shrink-0">Split at:</label>
                                        <input type="text" id="rt_msg_split" value=" - " class="w-20 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm text-center" oninput="updateMsgPreview()">
                                        <span class="text-[10px] text-gray-500">Delimiter to split into 2 tags</span>
                                    </div>

                                    <div class="grid grid-cols-2 gap-3">
                                        <div>
                                            <label class="text-xs text-orange-400 mb-1 block">Tag 1 Type</label>
                                            <select id="rt_msg_tag1_type_auto" class="w-full bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" onchange="updateMsgPreview()"></select>
                                        </div>
                                        <div>
                                            <label class="text-xs text-cyan-400 mb-1 block">Tag 2 Type</label>
                                            <select id="rt_msg_tag2_type_auto" class="w-full bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" onchange="updateMsgPreview()"></select>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Preview -->
                            <div class="bg-[#000] border border-[#333] rounded p-3">
                                <div class="flex justify-between items-center mb-2">
                                    <label class="text-xs text-gray-400 font-bold">PREVIEW</label>
                                    <div class="text-xs">
                                        <span id="rt_msg_char_count" class="text-green-400">0</span>
                                        <span class="text-gray-500">/</span>
                                        <span id="rt_msg_char_limit" class="text-gray-400">64</span>
                                        <span class="text-gray-600 ml-1">chars</span>
                                    </div>
                                </div>
                                <div id="rt_msg_preview" class="font-mono text-sm text-gray-300 min-h-[24px] whitespace-pre"></div>
                                <div class="mt-2 grid grid-cols-2 gap-2 text-xs" id="rt_msg_tag_info">
                                    <div class="text-orange-400">Tag 1: <span id="rt_msg_tag1_info" class="text-orange-300">-</span></div>
                                    <div class="text-cyan-400">Tag 2: <span id="rt_msg_tag2_info" class="text-cyan-300">-</span></div>
                                </div>
                                <div class="mt-1 font-mono text-[9px] text-gray-600 overflow-x-auto whitespace-nowrap">
                                    0----5----10---15---20---25---30---35---40---45---50---55---60---
                                </div>
                            </div>
                        </div>

                        <div class="rtplus-modal-footer">
                            <button onclick="closeRTMsgModal()" class="px-4 py-2 bg-[#333] hover:bg-[#444] rounded text-sm text-gray-300">Cancel</button>
                            <button onclick="saveRTMessage()" class="px-4 py-2 bg-[#d946ef] hover:bg-[#c026d3] rounded text-sm text-white font-bold">Save</button>
                        </div>
                    </div>
                </div>

                <!-- AF Pair Modal -->
                <div id="af_pair_modal" class="rtplus-modal-overlay" style="display: none;">
                    <div class="rtplus-modal-content" style="max-width: 500px;">
                        <div class="flex justify-between items-center mb-4">
                            <h3 id="af_pair_modal_title" class="text-lg font-bold">Add Frequency Pair</h3>
                            <button onclick="closeAFPairModal()" class="text-2xl leading-none hover:text-[#d946ef]">×</button>
                        </div>

                        <input type="hidden" id="af_pair_edit_id" value="">

                        <div class="space-y-4">
                            <!-- Main Frequency -->
                            <div>
                                <label class="text-xs text-gray-400 mb-1 block">Main Frequency (MHz)</label>
                                <input type="text" id="af_pair_main" class="w-full bg-[#111] border border-[#444] rounded px-2 py-1" placeholder="99.5">
                                <div class="text-[10px] text-gray-500 mt-1">The tuning frequency</div>
                            </div>

                            <!-- Alternative Frequencies -->
                            <div>
                                <label class="text-xs text-gray-400 mb-1 block">Alternative Frequencies (MHz)</label>
                                <textarea id="af_pair_alts" rows="3" class="w-full bg-[#111] border border-[#444] rounded px-2 py-1" placeholder="89.3, 101.7, 88.1"></textarea>
                                <div class="text-[10px] text-gray-500 mt-1">Comma-separated list of frequencies carrying same/regional variants</div>
                            </div>

                            <!-- Regional Variant Flag -->
                            <div class="flex items-center justify-between bg-[#111] border border-[#444] rounded px-3 py-2">
                                <div>
                                    <label class="text-xs text-gray-400">Regional Variant (RV)</label>
                                    <div class="text-[10px] text-gray-500 mt-1">Check if these frequencies carry different regional programming</div>
                                </div>
                                <input type="checkbox" class="toggle-checkbox" id="af_pair_regional">
                            </div>
                        </div>

                        <div class="flex justify-end gap-2 mt-6">
                            <button onclick="closeAFPairModal()" class="px-4 py-2 bg-[#444] hover:bg-[#555] rounded text-sm">Cancel</button>
                            <button onclick="saveAFPair()" class="px-4 py-2 bg-[#7c3aed] hover:bg-[#6d28d9] rounded text-sm text-white font-bold">Save</button>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-header">Flags & Switches</div>
                    <div class="section-body grid-cols-4">
                        <div class="flex flex-col items-center"><label>Traffic Prog</label><input type="checkbox" class="toggle-checkbox" id="tp" {% if state.tp %}checked{% endif %} onchange="sync()"></div>
                        <div class="flex flex-col items-center"><label>Traffic Ann</label><input type="checkbox" class="toggle-checkbox" id="ta" {% if state.ta %}checked{% endif %} onchange="sync()"></div>
                        <div class="flex flex-col items-center"><label>Music/Speech</label><input type="checkbox" class="toggle-checkbox" id="ms" {% if state.ms %}checked{% endif %} onchange="sync()"></div>
                        <div class="flex flex-col items-center"><label>Stereo</label><input type="checkbox" class="toggle-checkbox" id="di_stereo" {% if state.di_stereo %}checked{% endif %} onchange="sync()"></div>
                        <div class="flex flex-col items-center"><label>Artificial Head</label><input type="checkbox" class="toggle-checkbox" id="di_head" {% if state.di_head %}checked{% endif %} onchange="sync()"></div>
                        <div class="flex flex-col items-center"><label>Compressed</label><input type="checkbox" class="toggle-checkbox" id="di_comp" {% if state.di_comp %}checked{% endif %} onchange="sync()"></div>
                        <div class="flex flex-col items-center"><label>Dynamic PTY</label><input type="checkbox" class="toggle-checkbox" id="di_dyn" {% if state.di_dyn %}checked{% endif %} onchange="sync()"></div>
                    </div>
                </div>
                
                <div class="section">
                    <div class="section-header flex justify-between items-center">
                        <span>Alternative Frequencies (AF)</span>
                        <label class="flex items-center gap-2 cursor-pointer">
                            <span class="text-xs">Enable</span>
                            <input type="checkbox" class="toggle-checkbox" id="en_af" {% if state.en_af %}checked{% endif %} onchange="sync()">
                        </label>
                    </div>
                    <div class="section-body">
                         <div class="mb-3">
                             <label class="text-xs text-gray-400 mb-1 block">AF Method</label>
                             <select id="af_method" class="w-full" onchange="updateAFMethodUI(); sync();">
                                 <option value="A" {% if state.af_method == "A" %}selected{% endif %}>Method A - Simple list</option>
                                 <option value="B" {% if state.af_method == "B" %}selected{% endif %}>Method B - Frequency pairs</option>
                             </select>
                         </div>

                         <!-- Method A: Simple list -->
                         <div id="af_method_a_ui" style="display:none">
                             <label class="text-xs text-gray-400 mb-1 block">Frequency List (MHz)</label>
                             <textarea id="af_list" rows="2" placeholder="87.5, 98.1, 104.2" onchange="sync()">{{state.af_list}}</textarea>
                             <div class="text-[10px] text-gray-500 mt-1">Comma-separated list of all frequencies</div>
                         </div>

                         <!-- Method B: Frequency pairs -->
                         <div id="af_method_b_ui" style="display:none">
                             <div class="flex justify-between items-center mb-2">
                                 <span class="text-xs text-gray-400">Frequency Pairs</span>
                                 <button onclick="addAFPair()" class="px-2 py-1 bg-[#7c3aed] hover:bg-[#6d28d9] text-white text-xs rounded">+ Add Pair</button>
                             </div>
                             <div id="af_pairs_list" class="space-y-2"></div>
                             <div id="af_pairs_empty" class="text-center py-4 text-gray-600 text-sm" style="display:none">
                                 No frequency pairs configured. Click "+ Add Pair" to create one.
                             </div>
                         </div>
                    </div>
                </div>
            </div>

            <div id="expert" class="content">
                 <div class="section">
                    <div class="section-header">Scheduler Sequence</div>
                    <div class="section-body">
                         <div class="flex justify-between mb-2">
                             <label>Sequence String (e.g. 0A 0A 2A)</label>
                             <div class="flex items-center gap-2">
                                 <span class="text-[9px] text-gray-400">Manual / Auto</span>
                                 <input type="checkbox" class="toggle-checkbox" id="scheduler_auto" {% if state.scheduler_auto %}checked{% endif %} onchange="sync()">
                             </div>
                         </div>
                         <input type="text" id="group_sequence" value="{{state.group_sequence}}" onchange="sync()" class="font-mono">
                         <div class="text-[9px] text-gray-500 mt-1">
                             Auto Mode intelligently injects active groups (LPS, PTYN, ID, RT+) alongside standard PS/RT.
                         </div>
                    </div>
                </div>

                 <div class="section">
                     <div class="section-header">System IDs</div>
                     <div class="section-body grid-cols-2">
                         <div><label>ECC</label><input type="text" id="ecc" value="{{state.ecc}}" onchange="sync()"></div>
                         <div><label>LIC</label><input type="text" id="lic" value="{{state.lic}}" onchange="sync()"></div>
                         <div><label>Clock Offset</label><input type="number" id="tz_offset" value="{{state.tz_offset}}" onchange="sync()"></div>
                         <div><label>Enable CT</label><input type="checkbox" class="toggle-checkbox" id="en_ct" {% if state.en_ct %}checked{% endif %} onchange="sync()"></div>
                         <div><label>Enable ID (1A)</label><input type="checkbox" class="toggle-checkbox" id="en_id" {% if state.en_id %}checked{% endif %} onchange="sync()"></div>
                     </div>
                 </div>
                 
                 <div class="section">
                    <div class="section-header">Dynamic PTYN</div>
                    <div class="section-body">
                         <div class="flex justify-between">
                            <label>Enable</label>
                            <div class="flex gap-2 items-center"><label>Centre Text</label><input type="checkbox" class="toggle-checkbox" id="ptyn_centered" {% if state.ptyn_centered %}checked{% endif %} onchange="sync()"></div>
                             <input type="checkbox" class="toggle-checkbox" id="en_ptyn" {% if state.en_ptyn %}checked{% endif %} onchange="sync()">
                         </div>
                         <input type="text" id="ptyn" value="{{state.ptyn}}" onchange="sync()">
                    </div>
                 </div>

                 <div class="section">
                    <div class="section-header">Long PS (Group 15)</div>
                    <div class="section-body">
                         <div class="flex justify-between">
                             <label>Enable</label>
                             <div class="flex gap-2 items-center"><label>Centre Text</label><input type="checkbox" class="toggle-checkbox" id="lps_centered" {% if state.lps_centered %}checked{% endif %} onchange="sync()"></div>
                             <div class="flex gap-2 items-center"><label>Append CR</label><input type="checkbox" class="toggle-checkbox" id="lps_cr" {% if state.lps_cr %}checked{% endif %} onchange="sync()"></div>
                             <input type="checkbox" class="toggle-checkbox" id="en_lps" {% if state.en_lps %}checked{% endif %} onchange="sync()">
                         </div>
                         <input type="text" id="ps_long_32" value="{{state.ps_long_32}}" onchange="sync()">
                    </div>
                 </div>

                 <div class="section">
                    <div class="section-header">DAB Cross-Reference (Group 12A)</div>
                    <div class="section-body">
                         <div class="flex justify-between items-center mb-2">
                             <div>
                                 <label>Enable DAB Linkage</label>
                                 <div class="text-[9px] text-gray-500">Broadcast associated DAB ensemble frequency</div>
                                 <div class="text-[9px] text-amber-600 font-semibold">⚠️ EXPERIMENTAL: Not fully tested</div>
                             </div>
                             <input type="checkbox" class="toggle-checkbox" id="en_dab" {% if state.en_dab %}checked{% endif %} onchange="sync()">
                         </div>
                         <div class="mb-3">
                             <label>DAB Channel</label>
                             <select id="dab_channel" onchange="sync()">
                                 <option value="5A" {% if state.dab_channel == '5A' %}selected{% endif %}>5A (174.928 MHz)</option>
                                 <option value="5B" {% if state.dab_channel == '5B' %}selected{% endif %}>5B (176.640 MHz)</option>
                                 <option value="5C" {% if state.dab_channel == '5C' %}selected{% endif %}>5C (178.352 MHz)</option>
                                 <option value="5D" {% if state.dab_channel == '5D' %}selected{% endif %}>5D (180.064 MHz)</option>
                                 <option value="6A" {% if state.dab_channel == '6A' %}selected{% endif %}>6A (181.936 MHz)</option>
                                 <option value="6B" {% if state.dab_channel == '6B' %}selected{% endif %}>6B (183.648 MHz)</option>
                                 <option value="6C" {% if state.dab_channel == '6C' %}selected{% endif %}>6C (185.360 MHz)</option>
                                 <option value="6D" {% if state.dab_channel == '6D' %}selected{% endif %}>6D (187.072 MHz)</option>
                                 <option value="7A" {% if state.dab_channel == '7A' %}selected{% endif %}>7A (188.928 MHz)</option>
                                 <option value="7B" {% if state.dab_channel == '7B' %}selected{% endif %}>7B (190.640 MHz)</option>
                                 <option value="7C" {% if state.dab_channel == '7C' %}selected{% endif %}>7C (192.352 MHz)</option>
                                 <option value="7D" {% if state.dab_channel == '7D' %}selected{% endif %}>7D (194.064 MHz)</option>
                                 <option value="8A" {% if state.dab_channel == '8A' %}selected{% endif %}>8A (195.936 MHz)</option>
                                 <option value="8B" {% if state.dab_channel == '8B' %}selected{% endif %}>8B (197.648 MHz)</option>
                                 <option value="8C" {% if state.dab_channel == '8C' %}selected{% endif %}>8C (199.360 MHz)</option>
                                 <option value="8D" {% if state.dab_channel == '8D' %}selected{% endif %}>8D (201.072 MHz)</option>
                                 <option value="9A" {% if state.dab_channel == '9A' %}selected{% endif %}>9A (202.928 MHz)</option>
                                 <option value="9B" {% if state.dab_channel == '9B' %}selected{% endif %}>9B (204.640 MHz)</option>
                                 <option value="9C" {% if state.dab_channel == '9C' %}selected{% endif %}>9C (206.352 MHz)</option>
                                 <option value="9D" {% if state.dab_channel == '9D' %}selected{% endif %}>9D (208.064 MHz)</option>
                                 <option value="10A" {% if state.dab_channel == '10A' %}selected{% endif %}>10A (209.936 MHz)</option>
                                 <option value="10B" {% if state.dab_channel == '10B' %}selected{% endif %}>10B (211.648 MHz)</option>
                                 <option value="10C" {% if state.dab_channel == '10C' %}selected{% endif %}>10C (213.360 MHz)</option>
                                 <option value="10D" {% if state.dab_channel == '10D' %}selected{% endif %}>10D (215.072 MHz)</option>
                                 <option value="10N" {% if state.dab_channel == '10N' %}selected{% endif %}>10N (210.096 MHz)</option>
                                 <option value="11A" {% if state.dab_channel == '11A' %}selected{% endif %}>11A (216.928 MHz)</option>
                                 <option value="11B" {% if state.dab_channel == '11B' %}selected{% endif %}>11B (218.640 MHz)</option>
                                 <option value="11C" {% if state.dab_channel == '11C' %}selected{% endif %}>11C (220.352 MHz)</option>
                                 <option value="11D" {% if state.dab_channel == '11D' %}selected{% endif %}>11D (222.064 MHz)</option>
                                 <option value="11N" {% if state.dab_channel == '11N' %}selected{% endif %}>11N (217.088 MHz)</option>
                                 <option value="12A" {% if state.dab_channel == '12A' %}selected{% endif %}>12A (223.936 MHz)</option>
                                 <option value="12B" {% if state.dab_channel == '12B' %}selected{% endif %}>12B (225.648 MHz)</option>
                                 <option value="12C" {% if state.dab_channel == '12C' %}selected{% endif %}>12C (227.360 MHz)</option>
                                 <option value="12D" {% if state.dab_channel == '12D' %}selected{% endif %}>12D (229.072 MHz)</option>
                                 <option value="12N" {% if state.dab_channel == '12N' %}selected{% endif %}>12N (224.096 MHz)</option>
                                 <option value="13A" {% if state.dab_channel == '13A' %}selected{% endif %}>13A (230.784 MHz)</option>
                                 <option value="13B" {% if state.dab_channel == '13B' %}selected{% endif %}>13B (232.496 MHz)</option>
                                 <option value="13C" {% if state.dab_channel == '13C' %}selected{% endif %}>13C (234.208 MHz)</option>
                                 <option value="13D" {% if state.dab_channel == '13D' %}selected{% endif %}>13D (235.776 MHz)</option>
                                 <option value="13E" {% if state.dab_channel == '13E' %}selected{% endif %}>13E (237.488 MHz)</option>
                                 <option value="13F" {% if state.dab_channel == '13F' %}selected{% endif %}>13F (239.200 MHz)</option>
                             </select>
                         </div>
                         
                         <div class="grid grid-cols-2 gap-3 mb-3">
                             <div>
                                 <label>Ensemble ID (EId)</label>
                                 <div class="text-[9px] text-gray-500 mb-1">16-bit hex (e.g., CE15)</div>
                                 <input type="text" id="dab_eid" value="{{ state.dab_eid }}" maxlength="4" pattern="[0-9A-Fa-f]{4}" class="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded" onchange="sync()">
                             </div>
                             <div>
                                 <label>DAB Mode</label>
                                 <div class="text-[9px] text-gray-500 mb-1">Transmission mode</div>
                                 <select id="dab_mode" onchange="sync()" class="w-full">
                                     <option value="0" {% if state.dab_mode == 0 %}selected{% endif %}>Unspecified</option>
                                     <option value="1" {% if state.dab_mode == 1 %}selected{% endif %}>Mode I</option>
                                     <option value="2" {% if state.dab_mode == 2 %}selected{% endif %}>Mode II/III</option>
                                     <option value="3" {% if state.dab_mode == 3 %}selected{% endif %}>Mode IV</option>
                                 </select>
                             </div>
                         </div>
                         
                         <div class="grid grid-cols-2 gap-3 mb-3">
                             <div>
                                 <label>Table Type (E/S Flag)</label>
                                 <div class="text-[9px] text-gray-500 mb-1">Ensemble or Service table</div>
                                 <select id="dab_es_flag" onchange="sync()" class="w-full">
                                     <option value="0" {% if state.dab_es_flag == 0 %}selected{% endif %}>Ensemble (0)</option>
                                     <option value="1" {% if state.dab_es_flag == 1 %}selected{% endif %}>Service (1)</option>
                                 </select>
                             </div>
                             <div>
                                 <label>Service ID (SId)</label>
                                 <div class="text-[9px] text-gray-500 mb-1">16-bit hex (for service table)</div>
                                 <input type="text" id="dab_sid" value="{{ state.dab_sid }}" maxlength="4" pattern="[0-9A-Fa-f]{4}" class="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded" onchange="sync()">
                             </div>
                         </div>
                         
                         <div>
                             <label>Variant Code</label>
                             <div class="text-[9px] text-gray-500 mb-1">For service table format</div>
                             <select id="dab_variant" onchange="sync()" class="w-full">
                                 <option value="0" {% if state.dab_variant == 0 %}selected{% endif %}>0 - Ensemble information</option>
                                 <option value="1" {% if state.dab_variant == 1 %}selected{% endif %}>1 - Linkage information</option>
                             </select>
                         </div>
                    </div>
                 </div>

                 <div class="section">
                    <div class="section-header">EON - Enhanced Other Networks (Group 14A)</div>
                    <div class="section-body">
                         <div class="flex justify-between items-center mb-3">
                             <div>
                                 <label>Enable EON</label>
                                 <div class="text-[9px] text-gray-500">Broadcast information about other radio stations</div>
                             </div>
                             <input type="checkbox" class="toggle-checkbox" id="en_eon" {% if state.en_eon %}checked{% endif %} onchange="sync()">
                         </div>

                         <input type="hidden" id="eon_services" value="{{state.eon_services}}">

                         <div class="mb-3">
                             <button onclick="openEONModal()" class="bg-blue-600 hover:bg-blue-500 text-white rounded px-3 py-2 text-sm w-full">Manage EON Services</button>
                         </div>

                         <div id="eon_services_display" class="text-xs text-gray-400">
                             No services configured
                         </div>
                    </div>
                 </div>
            </div>

            <div id="audio" class="content">
                <div class="section border-l-4 border-l-orange-500">
                    <div class="section-header text-orange-400">Audio I/O & Genlock</div>
                    <div class="section-body">
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label>Input Device (MPX or Audio)</label>
                                <select id="dev_in">
                                    <option value="-1">None (Internal Generator Only)</option>
                                    {% for d in inputs %}<option value="{{d.index}}" {% if d.index == state.device_in_idx %}selected{% endif %}>{{d.name}}</option>{% endfor %}
                                </select>
                            </div>
                            <div>
                                <label>Output Device (Transmitter)</label>
                                <select id="dev_out">
                                    {% for d in outputs %}<option value="{{d.index}}" {% if d.index == state.device_out_idx %}selected{% endif %}>{{d.name}}</option>{% endfor %}
                                </select>
                            </div>
                        </div>
                        
                        <div class="grid grid-cols-2 gap-4 mt-2">
                            <div class="bg-[#111] p-2 rounded flex justify-between items-center">
                                <div>
                                    <label class="text-orange-400">Pass-through</label>
                                    <div class="text-[9px] text-gray-500">Mix input audio to output</div>
                                </div>
                                <input type="checkbox" class="toggle-checkbox" id="passthrough" {% if state.passthrough %}checked{% endif %} onchange="sync()">
                            </div>
                            <div class="bg-[#111] p-2 rounded flex justify-between items-center">
                                <div>
                                    <label class="text-orange-400">Genlock (FFT Sync)</label>
                                    <div class="text-[9px] text-gray-500">Auto-lock RDS to Input 19k</div>
                                </div>
                                <input type="checkbox" class="toggle-checkbox" id="genlock" {% if state.genlock %}checked{% endif %} onchange="sync()">
                            </div>
                        </div>
                        
                        <div class="mt-2">
                            <label>Genlock Phase Offset (Degrees)</label>
                            <div class="slider-container">
                                <input type="range" id="genlock_offset" min="0" max="360" value="{{state.genlock_offset}}" oninput="sync()">
                                <span class="slider-val" id="val_offset">{{state.genlock_offset}}</span>
                            </div>
                        </div>

                        <div class="mt-2">
                            <label>RDS Carrier Frequency (Hz)</label>
                            <div class="grid grid-cols-2 gap-4 items-center">
                                <input type="number" id="rds_freq" value="{{state.rds_freq}}" min="1000" max="120000" step="1" class="w-32" onchange="sync()">
                                <div class="text-[11px] {{ 'text-red-400' if state.rds_freq != 57000 else 'text-gray-400' }}">
                                    ⚠ Non-compliant if not 57000 Hz. Use at your own risk.
                                </div>
                            </div>
                        </div>

                        <div class="grid grid-cols-2 gap-4 mt-2">
                            <div>
                                <label>19kHz Pilot Level (%)</label>
                                <div class="slider-container">
                                    <input type="range" id="pilot_level" min="0" max="20" step="0.1" value="{{state.pilot_level}}" oninput="sync()">
                                    <span class="slider-val" id="val_pilot">{{state.pilot_level}}</span>
                                </div>
                                <div class="text-[11px] text-gray-400" id="pilot_note">Pilot will be disabled when pass-through is active with an input device</div>
                                <div class="text-[11px] text-gray-300 mt-1">Status: <span id="pilot_status">Unknown</span></div>
                            </div>
                            <div>
                                <label>RDS Carrier Level (%)</label>
                                <div class="slider-container">
                                    <input type="range" id="rds_level" min="0" max="20" step="0.1" value="{{state.rds_level}}" oninput="sync()">
                                    <span class="slider-val" id="val_rds">{{state.rds_level}}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="datasets" class="content">
                <div class="section">
                    <div class="section-header">RDS Datasets</div>
                    <div class="section-body">
                        <p class="text-sm text-gray-400 mb-4">Save and switch between different RDS configurations. Each dataset stores all RDS settings.</p>

                        <div class="grid grid-cols-3 gap-3 mb-4" id="dataset_buttons">
                        </div>

                        <div class="flex gap-2">
                            <button onclick="createDataset()" class="bg-green-600 hover:bg-green-500 text-white rounded px-3 py-2 text-sm">+ New Dataset</button>
                            <button onclick="renameCurrentDataset()" class="bg-blue-600 hover:bg-blue-500 text-white rounded px-3 py-2 text-sm">Rename</button>
                            <button onclick="deleteCurrentDataset()" class="bg-red-600 hover:bg-red-500 text-white rounded px-3 py-2 text-sm">Delete</button>
                        </div>

                        <div class="mt-4 p-3 bg-black/40 rounded border border-gray-700">
                            <div class="text-xs text-gray-400">
                                <strong>Current Dataset:</strong> <span id="current_dataset_name">Dataset 1</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div id="settings" class="content">
                <div class="section">
                    <div class="section-header">Startup & Access</div>
                    <div class="section-body">
                        <div class="flex items-center justify-between mb-3">
                            <div>
                                <label>Auto-start encoder on launch</label>
                                <div class="text-[9px] text-gray-500">Default is enabled; uncheck to keep the encoder idle on boot.</div>
                            </div>
                            <input type="checkbox" class="toggle-checkbox" id="auto_start" {% if state.auto_start %}checked{% endif %}>
                        </div>
                        <div class="mb-2">
                            <label>Username</label>
                            <input type="text" id="auth_user" value="{{ auth_config.get('user','') }}" class="w-full bg-black/60 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-pink-500">
                        </div>
                        <div class="mb-3">
                            <label>New Password</label>
                            <input type="password" id="auth_pass" placeholder="Leave blank to keep current" class="w-full bg-black/60 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-pink-500">
                        </div>
                        <div class="flex gap-2 items-center">
                            <button onclick="saveSettings()" class="bg-pink-600 hover:bg-pink-500 text-white font-semibold rounded px-4 py-2 text-sm transition">Save Settings</button>
                            <a href="/logout" class="text-[12px] text-gray-300 hover:text-white underline">Logout</a>
                        </div>
                        <div id="settings_status" class="text-[11px] text-gray-400 mt-2"></div>
                    </div>
                </div>
            </div>

        </div>
    </div>

    <!-- RT+ Builder Modal -->
    <div id="rtplus_modal" class="rtplus-modal-overlay" style="display: none;">
        <div class="rtplus-modal">
            <div class="rtplus-modal-header">
                <div>
                    <div class="text-lg font-bold text-white">RT+ Message Builder</div>
                    <div class="text-xs text-gray-400">Build RadioText with tagged content (supports all 64 RT+ content types)</div>
                </div>
                <button onclick="closeRTPlusModal()" class="text-gray-400 hover:text-white text-2xl leading-none">&times;</button>
            </div>

            <div class="rtplus-modal-body space-y-4">
                <!-- Buffer Tabs (only shown in manual buffer mode) -->
                <div id="builder_buffer_tabs" class="flex gap-2" style="display: none;">
                    <button id="builder_tab_a" onclick="setBuilderBuffer('a')" class="px-4 py-2 rounded font-bold bg-[#d946ef] text-white">Buffer A</button>
                    <button id="builder_tab_b" onclick="setBuilderBuffer('b')" class="px-4 py-2 rounded font-bold bg-[#333] text-gray-400 hover:bg-[#444]">Buffer B</button>
                </div>

                <!-- Split Mode -->
                <div class="bg-[#252525] border border-[#333] rounded p-3">
                    <div class="flex items-center justify-between mb-2">
                        <div>
                            <label class="text-xs text-gray-400 font-bold">SPLIT MODE</label>
                            <div class="text-[10px] text-gray-500">Parse existing RT text into tagged segments</div>
                        </div>
                        <button onclick="showSplitPanel()" id="btn_show_split" class="px-3 py-1 bg-[#333] hover:bg-[#444] rounded text-xs text-gray-300">Split Text...</button>
                    </div>
                    <div id="split_panel" class="hidden mt-3 space-y-2 border-t border-[#444] pt-3">
                        <div class="flex gap-2 items-center">
                            <label class="w-20 text-xs text-gray-500 shrink-0">Source:</label>
                            <input type="text" id="split_source" class="flex-1 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" placeholder="Artist Name - Song Title">
                        </div>
                        <div class="flex gap-2 items-center">
                            <label class="w-20 text-xs text-gray-500 shrink-0">Split at:</label>
                            <input type="text" id="split_delimiter" class="w-24 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm text-center" value=" - ">
                            <button onclick="performSplit()" class="px-3 py-1 bg-[#d946ef] hover:bg-[#c026d3] rounded text-xs text-white font-bold">Split</button>
                            <button onclick="hideSplitPanel()" class="px-3 py-1 bg-[#333] hover:bg-[#444] rounded text-xs text-gray-300">Cancel</button>
                        </div>
                        <div class="text-[10px] text-gray-500">Tip: Use \r"path" or \w"url" in source for dynamic content</div>
                    </div>
                </div>

                <!-- Message Construction -->
                <div class="bg-[#252525] border border-[#333] rounded p-4 space-y-3">
                    <div class="text-xs text-gray-400 mb-2">Build your message with up to 2 tagged segments:</div>

                    <!-- Prefix -->
                    <div class="flex gap-2 items-center">
                        <label class="w-20 text-xs text-gray-500 shrink-0">Prefix:</label>
                        <input type="text" id="builder_prefix" class="flex-1 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" placeholder="Text before first tag" oninput="updateBuilderPreview()">
                    </div>

                    <!-- Tag 1 -->
                    <div class="bg-[#1a1a1a] border border-orange-900/50 rounded p-3 space-y-2">
                        <div class="flex gap-2 items-center">
                            <label class="w-20 text-xs text-orange-400 font-bold shrink-0">Tag 1:</label>
                            <select id="builder_tag1_type" class="flex-1 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" onchange="updateBuilderPreview()"></select>
                        </div>
                        <div class="flex gap-2 items-center">
                            <label class="w-20 text-xs text-gray-500 shrink-0">Content:</label>
                            <input type="text" id="builder_tag1_text" class="flex-1 bg-[#111] border border-orange-900/50 rounded px-2 py-1 text-sm text-orange-300" placeholder="Text or \r&quot;path&quot; for file" oninput="updateBuilderPreview()">
                        </div>
                        <div class="text-[10px] text-gray-500">Supports: \r"path" (file), \R"path" (file uppercase), \w"url" (web)</div>
                    </div>

                    <!-- Middle -->
                    <div class="flex gap-2 items-center">
                        <label class="w-20 text-xs text-gray-500 shrink-0">Between:</label>
                        <input type="text" id="builder_middle" class="flex-1 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" placeholder=" - " value=" - " oninput="updateBuilderPreview()">
                    </div>

                    <!-- Tag 2 -->
                    <div class="bg-[#1a1a1a] border border-cyan-900/50 rounded p-3 space-y-2">
                        <div class="flex gap-2 items-center">
                            <label class="w-20 text-xs text-cyan-400 font-bold shrink-0">Tag 2:</label>
                            <select id="builder_tag2_type" class="flex-1 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" onchange="updateBuilderPreview()"></select>
                        </div>
                        <div class="flex gap-2 items-center">
                            <label class="w-20 text-xs text-gray-500 shrink-0">Content:</label>
                            <input type="text" id="builder_tag2_text" class="flex-1 bg-[#111] border border-cyan-900/50 rounded px-2 py-1 text-sm text-cyan-300" placeholder="Text or \r&quot;path&quot; for file" oninput="updateBuilderPreview()">
                        </div>
                        <div class="text-[10px] text-gray-500">Supports: \r"path" (file), \R"path" (file uppercase), \w"url" (web)</div>
                    </div>

                    <!-- Suffix -->
                    <div class="flex gap-2 items-center">
                        <label class="w-20 text-xs text-gray-500 shrink-0">Suffix:</label>
                        <input type="text" id="builder_suffix" class="flex-1 bg-[#111] border border-[#444] rounded px-2 py-1 text-sm" placeholder="Text after tags" oninput="updateBuilderPreview()">
                    </div>
                </div>

                <!-- Preview -->
                <div class="bg-[#000] border border-[#333] rounded p-4">
                    <div class="flex justify-between items-center mb-2">
                        <label class="text-xs text-gray-400 font-bold">PREVIEW</label>
                        <div class="text-xs">
                            <span id="builder_char_count" class="text-green-400">0</span>
                            <span class="text-gray-500">/</span>
                            <span id="builder_char_limit" class="text-gray-400">64</span>
                            <span class="text-gray-600 ml-1">chars</span>
                        </div>
                    </div>
                    <div id="builder_preview" class="rtplus-preview text-gray-300"></div>
                    <div class="mt-3 grid grid-cols-2 gap-2 text-xs">
                        <div class="text-orange-400">Tag 1: <span id="builder_tag1_info" class="text-orange-300">-</span></div>
                        <div class="text-orange-400">Type: <span id="builder_tag1_typename" class="text-orange-300">-</span></div>
                        <div class="text-cyan-400">Tag 2: <span id="builder_tag2_info" class="text-cyan-300">-</span></div>
                        <div class="text-cyan-400">Type: <span id="builder_tag2_typename" class="text-cyan-300">-</span></div>
                    </div>
                    <div class="mt-2 font-mono text-[9px] text-gray-600 overflow-x-auto whitespace-nowrap">
                        0----5----10---15---20---25---30---35---40---45---50---55---60---
                    </div>
                </div>

                <!-- Mode Toggle -->
                <div class="flex items-center gap-4 bg-[#252525] border border-[#333] rounded p-3">
                    <span class="text-xs text-gray-400">RT+ Mode:</span>
                    <label class="flex items-center gap-2 cursor-pointer">
                        <input type="radio" name="rtplus_mode" value="format" class="accent-[#d946ef]" onchange="updateRTPlusMode()">
                        <span class="text-sm text-gray-300">Format String</span>
                        <span class="text-[10px] text-gray-500">(legacy)</span>
                    </label>
                    <label class="flex items-center gap-2 cursor-pointer">
                        <input type="radio" name="rtplus_mode" value="builder" class="accent-[#d946ef]" onchange="updateRTPlusMode()">
                        <span class="text-sm text-gray-300">Builder</span>
                        <span class="text-[10px] text-gray-500">(new)</span>
                    </label>
                </div>
            </div>

            <div class="rtplus-modal-footer">
                <button onclick="resetBuilder()" class="px-3 py-2 bg-[#333] hover:bg-[#444] rounded text-sm text-gray-300">Reset</button>
                <div class="flex gap-2">
                    <button onclick="closeRTPlusModal()" class="px-4 py-2 bg-[#333] hover:bg-[#444] rounded text-sm text-gray-300">Cancel</button>
                    <button onclick="applyBuilderConfig()" class="px-4 py-2 bg-[#d946ef] hover:bg-[#c026d3] rounded text-sm text-white font-bold">Apply</button>
                </div>
            </div>
        </div>
    </div>

    <div id="eon_modal" class="rtplus-modal-overlay" style="display: none;">
        <div class="rtplus-modal-content" style="max-width: 600px;">
            <div class="flex justify-between items-center mb-4">
                <h3 id="eon_modal_title" class="text-lg font-bold">Manage EON Services</h3>
                <button onclick="closeEONModal()" class="text-2xl leading-none hover:text-pink-600">×</button>
            </div>

            <div class="mb-4">
                <div id="eon_service_list" class="space-y-2 mb-3">
                </div>
                <button onclick="addEONService()" class="bg-green-600 hover:bg-green-500 text-white rounded px-3 py-2 text-sm w-full">+ Add Service</button>
            </div>

            <div id="eon_edit_form" style="display: none;" class="border-t border-gray-700 pt-4 mt-4">
                <input type="hidden" id="eon_edit_idx" value="">

                <div class="space-y-3">
                    <div>
                        <label class="text-xs text-gray-400 mb-1 block">PI Code (ON)</label>
                        <input type="text" id="eon_pi" maxlength="4" pattern="[0-9A-Fa-f]{4}" class="w-full bg-black border border-gray-600 rounded px-2 py-1 font-mono" placeholder="C201">
                        <div class="text-[10px] text-gray-500 mt-1">4-digit hex code</div>
                    </div>

                    <div>
                        <label class="text-xs text-gray-400 mb-1 block">PS Name (ON)</label>
                        <input type="text" id="eon_ps" maxlength="8" class="w-full bg-black border border-gray-600 rounded px-2 py-1 font-mono" placeholder="OTHER FM">
                        <div class="text-[10px] text-gray-500 mt-1">8 characters max</div>
                    </div>

                    <div>
                        <label class="text-xs text-gray-400 mb-1 block">PTY (ON)</label>
                        <select id="eon_pty" class="w-full bg-black border border-gray-600 rounded px-2 py-1">
                            <option value="0">None</option>
                            <option value="1">News</option>
                            <option value="2">Current Affairs</option>
                            <option value="3">Information</option>
                            <option value="4">Sport</option>
                            <option value="5">Education</option>
                            <option value="6">Drama</option>
                            <option value="7">Culture</option>
                            <option value="8">Science</option>
                            <option value="9">Varied</option>
                            <option value="10">Pop Music</option>
                            <option value="11">Rock Music</option>
                            <option value="12">Easy Listening</option>
                            <option value="13">Light Classical</option>
                            <option value="14">Serious Classical</option>
                            <option value="15">Other Music</option>
                        </select>
                    </div>

                    <div>
                        <label class="text-xs text-gray-400 mb-1 block">AF List (ON)</label>
                        <input type="text" id="eon_af" class="w-full bg-black border border-gray-600 rounded px-2 py-1" placeholder="88.1, 101.5">
                        <div class="text-[10px] text-gray-500 mt-1">Comma-separated frequencies in MHz</div>
                    </div>

                    <div class="grid grid-cols-2 gap-3">
                        <div class="flex items-center justify-between bg-black p-2 rounded">
                            <label class="text-xs text-gray-400">TP (ON)</label>
                            <input type="checkbox" class="toggle-checkbox" id="eon_tp">
                        </div>
                        <div class="flex items-center justify-between bg-black p-2 rounded">
                            <label class="text-xs text-gray-400">TA (ON)</label>
                            <input type="checkbox" class="toggle-checkbox" id="eon_ta">
                        </div>
                    </div>
                </div>

                <div class="flex justify-end gap-2 mt-4">
                    <button onclick="cancelEONEdit()" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm">Cancel</button>
                    <button onclick="saveEONService()" class="px-4 py-2 bg-purple-600 hover:bg-purple-500 rounded text-sm text-white font-bold">Save</button>
                </div>
            </div>

            <div class="flex justify-end mt-4">
                <button onclick="closeEONModal()" class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm">Close</button>
            </div>
        </div>
    </div>

    <script>
        var socket = io();
        var running = {{ 'true' if state.running else 'false' }};
        var pty_list_rds = {{ pty_list_rds|tojson }};
        var pty_list_rbds = {{ pty_list_rbds|tojson }};

        // RT+ Content Types Dictionary
        var RTPLUS_TYPES = {
            0: ["Dummy", "No content type"],
            1: ["Title", "Item title"],
            2: ["Album", "Album/CD name"],
            3: ["Track", "Track number"],
            4: ["Artist", "Artist name"],
            5: ["Composition", "Composition name"],
            6: ["Movement", "Movement name"],
            7: ["Conductor", "Conductor"],
            8: ["Composer", "Composer"],
            9: ["Band", "Band/Orchestra"],
            10: ["Comment", "Free text comment"],
            11: ["Genre", "Genre"],
            12: ["News", "News headlines"],
            13: ["News.Local", "Local news"],
            14: ["Stock", "Stock market"],
            15: ["Sport", "Sport news"],
            16: ["Lottery", "Lottery numbers"],
            17: ["Horoscope", "Horoscope"],
            18: ["Daily", "Daily diversion"],
            19: ["Health", "Health tips"],
            20: ["Event", "Event info"],
            21: ["Scene", "Scene/Film info"],
            22: ["Cinema", "Cinema info"],
            23: ["TV", "TV info"],
            24: ["DateTime", "Date/Time"],
            25: ["Weather", "Weather info"],
            26: ["Traffic", "Traffic info"],
            27: ["Alarm", "Alarm/Emergency"],
            28: ["Advert", "Advertisement"],
            29: ["URL", "Website URL"],
            30: ["Other", "Other info"],
            31: ["Stn.Short", "Station name short"],
            32: ["Stn.Long", "Station name long"],
            33: ["Prog.Now", "Current program"],
            34: ["Prog.Next", "Next program"],
            35: ["Prog.Part", "Program part"],
            36: ["Host", "Host name"],
            37: ["Editorial", "Editorial staff"],
            38: ["Frequency", "Frequency info"],
            39: ["Homepage", "Homepage URL"],
            40: ["Subchannel", "Sub-channel"],
            41: ["Phone.Hotline", "Hotline phone"],
            42: ["Phone.Studio", "Studio phone"],
            43: ["Phone.Other", "Other phone"],
            44: ["SMS.Studio", "Studio SMS"],
            45: ["SMS.Other", "Other SMS"],
            46: ["Email.Hotline", "Hotline email"],
            47: ["Email.Studio", "Studio email"],
            48: ["Email.Other", "Other email"],
            49: ["MMS.Phone", "MMS number"],
            50: ["Chat", "Chat"],
            51: ["Chat.Centre", "Chat centre"],
            52: ["Vote.Question", "Vote question"],
            53: ["Vote.Centre", "Vote centre"],
            54: ["RFU", "Reserved"],
            55: ["RFU", "Reserved"],
            56: ["RFU", "Reserved"],
            57: ["RFU", "Reserved"],
            58: ["RFU", "Reserved"],
            59: ["Place", "Place/Location"],
            60: ["Appointment", "Appointment"],
            61: ["Identifier", "Identifier"],
            62: ["Purchase", "Purchase info"],
            63: ["GetData", "Get Data"]
        };

        // RT Messages State
        var rtMessages = [];
        try {
            var rtMessagesStr = {{ state.rt_messages|tojson }};
            rtMessages = JSON.parse(rtMessagesStr) || [];
        } catch(e) {
            console.error("Failed to parse rt_messages:", e);
            rtMessages = [];
        }
        var currentMsgBuffer = 'AB';

        // RT+ Builder State (legacy)
        var currentBuilderBuffer = 'a';
        var builderState = {
            a: { prefix: '', tag1_type: 4, tag1_text: '', middle: ' - ', tag2_type: 1, tag2_text: '', suffix: '' },
            b: { prefix: '', tag1_type: 4, tag1_text: '', middle: ' - ', tag2_type: 1, tag2_text: '', suffix: '' }
        };

        // === RT MESSAGE FUNCTIONS ===
        function renderRTMessages() {
            var container = document.getElementById('rt_messages_list');
            var emptyState = document.getElementById('rt_messages_empty');

            if (!rtMessages || rtMessages.length === 0) {
                container.innerHTML = '';
                emptyState.style.display = 'block';
                return;
            }

            emptyState.style.display = 'none';
            container.innerHTML = rtMessages.map(function(msg, idx) {
                return renderMessageCard(msg, idx);
            }).join('');

            // Attach drag-and-drop handlers after rendering
            attachDragHandlers();
        }

        var draggedIndex = -1;

        function attachDragHandlers() {
            var cards = document.querySelectorAll('.rt-msg-card');
            cards.forEach(function(card, idx) {
                card.draggable = true;

                card.addEventListener('dragstart', function(e) {
                    draggedIndex = idx;
                    card.classList.add('dragging');
                    e.dataTransfer.effectAllowed = 'move';
                });

                card.addEventListener('dragend', function(e) {
                    card.classList.remove('dragging');
                    // Remove all drag-over classes
                    document.querySelectorAll('.rt-msg-card').forEach(function(c) {
                        c.classList.remove('drag-over');
                    });
                });

                card.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';

                    if (idx !== draggedIndex) {
                        card.classList.add('drag-over');
                    }
                });

                card.addEventListener('dragleave', function(e) {
                    card.classList.remove('drag-over');
                });

                card.addEventListener('drop', function(e) {
                    e.preventDefault();
                    card.classList.remove('drag-over');

                    if (idx !== draggedIndex && draggedIndex >= 0) {
                        // Reorder array
                        var draggedMsg = rtMessages[draggedIndex];
                        rtMessages.splice(draggedIndex, 1);
                        rtMessages.splice(idx, 0, draggedMsg);

                        // Re-render and sync
                        renderRTMessages();
                        syncRTMessages();
                    }
                });
            });
        }

        function renderMessageCard(msg, idx) {
            var bufferClass = 'buffer-' + msg.buffer.toLowerCase();
            var sourceIcon = {manual: '✏️', file: '📄', url: '🌐'}[msg.source_type] || '✏️';
            var preview = (msg.content || '').substring(0, 50) + ((msg.content || '').length > 50 ? '...' : '');
            var enabledClass = msg.enabled ? '' : 'disabled';

            return '<div class="rt-msg-card ' + enabledClass + '" data-id="' + msg.id + '">' +
                '<div class="rt-msg-header" onclick="editRTMessage(\'' + msg.id + '\')">' +
                '<span class="rt-msg-buffer ' + bufferClass + '">' + msg.buffer + '</span>' +
                '<span class="rt-msg-duration">' + (msg.cycles || 2) + ' cycles</span>' +
                '<span class="rt-msg-source">' + sourceIcon + '</span>' +
                '<div class="rt-msg-preview">' + escapeHtml(preview) + '</div>' +
                (msg.rt_plus_enabled ? '<span class="rt-msg-rtplus">RT+</span>' : '') +
                '<div class="rt-msg-actions">' +
                '<button onclick="event.stopPropagation(); toggleRTMessage(\'' + msg.id + '\')" title="' + (msg.enabled ? 'Disable' : 'Enable') + '">' + (msg.enabled ? '👁' : '👁‍🗨') + '</button>' +
                '<button onclick="event.stopPropagation(); deleteRTMessage(\'' + msg.id + '\')" title="Delete">×</button>' +
                '</div></div></div>';
        }

        function escapeHtml(text) {
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        var pendingNewMessage = null; // Track message being created

        function addRTMessage() {
            pendingNewMessage = {
                id: 'msg_' + Date.now(),
                buffer: 'AB',
                cycles: 2,
                source_type: 'manual',
                content: '',
                split_delimiter: ' - ',
                rt_plus_enabled: false,
                rt_plus_tags: { tag1_type: 4, tag2_type: 1 },
                enabled: true,
                prefix: '',
                tag1_text: '',
                middle: ' - ',
                tag2_text: '',
                suffix: ''
            };
            openMessageModal(pendingNewMessage, true);
        }

        function openMessageModal(msg, isNew) {
            document.getElementById('rt_msg_edit_id').value = msg.id;
            document.getElementById('rt_msg_modal_title').textContent = isNew ? 'Add Message' : 'Edit Message';
            document.getElementById('rt_msg_cycles').value = msg.cycles || 2;

            // Set source type radio first
            var sourceRadios = document.querySelectorAll('input[name="rt_msg_source"]');
            sourceRadios.forEach(function(r) { r.checked = (r.value === (msg.source_type || 'manual')); });

            // Set buffer buttons
            currentMsgBuffer = msg.buffer || 'AB';
            updateBufferButtons();

            // Set RT+ enabled toggle
            document.getElementById('rt_msg_rtplus_enabled').checked = msg.rt_plus_enabled || false;

            // Load fields based on source type
            if (msg.source_type === 'manual') {
                if (msg.content && !msg.rt_plus_enabled) {
                    document.getElementById('rt_msg_simple_text').value = msg.content || '';
                } else {
                    document.getElementById('rt_msg_prefix').value = msg.prefix || '';
                    document.getElementById('rt_msg_tag1_text').value = msg.tag1_text || '';
                    document.getElementById('rt_msg_middle').value = msg.middle || ' - ';
                    document.getElementById('rt_msg_tag2_text').value = msg.tag2_text || '';
                    document.getElementById('rt_msg_suffix').value = msg.suffix || '';
                    document.getElementById('rt_msg_tag1_type').value = (msg.rt_plus_tags && msg.rt_plus_tags.tag1_type) || 4;
                    document.getElementById('rt_msg_tag2_type').value = (msg.rt_plus_tags && msg.rt_plus_tags.tag2_type) || 1;
                }
            } else {
                document.getElementById('rt_msg_content').value = msg.content || '';
                document.getElementById('rt_msg_prefix_auto').value = msg.prefix || '';
                document.getElementById('rt_msg_suffix_auto').value = msg.suffix || '';
                document.getElementById('rt_msg_split').value = msg.split_delimiter || ' - ';
                document.getElementById('rt_msg_tag1_type_auto').value = (msg.rt_plus_tags && msg.rt_plus_tags.tag1_type) || 4;
                document.getElementById('rt_msg_tag2_type_auto').value = (msg.rt_plus_tags && msg.rt_plus_tags.tag2_type) || 1;
            }

            updateSourceUI();
            updateRTPlusUI();
            updateMsgPreview();
            document.getElementById('rt_msg_modal').style.display = 'flex';
        }

        function editRTMessage(id) {
            var msg = rtMessages.find(function(m) { return m.id === id; });
            if (!msg) return;
            pendingNewMessage = null; // Clear any pending new message
            openMessageModal(msg, false);
        }

        function closeRTMsgModal() {
            pendingNewMessage = null; // Clear pending message if modal closed without saving
            document.getElementById('rt_msg_modal').style.display = 'none';
        }

        function setMsgBuffer(buf) {
            currentMsgBuffer = buf;
            updateBufferButtons();
        }

        function updateBufferButtons() {
            ['A', 'B', 'AB'].forEach(function(b) {
                var btn = document.getElementById('rt_msg_buf_' + b.toLowerCase());
                if (b === currentMsgBuffer) {
                    btn.className = 'px-4 py-2 rounded font-bold text-sm ' + (b === 'A' ? 'bg-[#dc2626]' : b === 'B' ? 'bg-[#2563eb]' : 'bg-[#7c3aed]') + ' text-white';
                } else {
                    btn.className = 'px-4 py-2 rounded font-bold text-sm bg-[#333] text-gray-400 hover:bg-[#444]';
                }
            });
        }

        function updateSourceUI() {
            var sourceType = document.querySelector('input[name="rt_msg_source"]:checked').value;
            var contentWrap = document.getElementById('rt_msg_content_wrap');
            var manualSimple = document.getElementById('rt_msg_manual_simple');
            var manualBuilder = document.getElementById('rt_msg_manual_builder');
            var rtplusOptions = document.getElementById('rt_msg_rtplus_options');
            var tagInfo = document.getElementById('rt_msg_tag_info');

            if (sourceType === 'manual') {
                // Show manual input (simple or builder depending on RT+ toggle)
                contentWrap.style.display = 'none';
                updateRTPlusUI(); // This will handle showing simple vs builder
                tagInfo.style.display = 'grid';
            } else {
                // Show content input, hide manual inputs
                contentWrap.style.display = 'block';
                if (manualSimple) manualSimple.style.display = 'none';
                if (manualBuilder) manualBuilder.style.display = 'none';
                tagInfo.style.display = 'grid';

                var label = document.getElementById('rt_msg_content_label');
                var hint = document.getElementById('rt_msg_content_hint');

                if (sourceType === 'file') {
                    label.textContent = 'Content Source';
                    hint.textContent = 'Full path to text file (e.g., C:\\nowplaying.txt)';
                } else if (sourceType === 'url') {
                    label.textContent = 'Content Source';
                    hint.textContent = 'URL to fetch text from';
                }
            }
            updateRTPlusUI();
            updateMsgPreview();
        }

        function updateRTPlusUI() {
            var enabled = document.getElementById('rt_msg_rtplus_enabled').checked;
            var sourceType = document.querySelector('input[name="rt_msg_source"]:checked').value;
            var opts = document.getElementById('rt_msg_rtplus_options');
            var manualSimple = document.getElementById('rt_msg_manual_simple');
            var manualBuilder = document.getElementById('rt_msg_manual_builder');

            // For manual mode: show simple text input or RT+ builder based on toggle
            if (sourceType === 'manual') {
                if (manualSimple) manualSimple.style.display = enabled ? 'none' : 'block';
                if (manualBuilder) manualBuilder.style.display = enabled ? 'block' : 'none';
                if (opts) opts.style.display = 'none';
            } else {
                // For file/URL: hide manual inputs, show RT+ options if enabled
                if (manualSimple) manualSimple.style.display = 'none';
                if (manualBuilder) manualBuilder.style.display = 'none';
                if (opts) opts.style.display = enabled ? 'block' : 'none';
            }
            updateMsgPreview();
        }

        // Cache for resolved content
        var resolvedContentCache = '';
        var resolveDebounceTimer = null;
        var lastResolvePath = '';
        var lastResolveType = '';

        function fetchResolvedContent() {
            var sourceType = document.querySelector('input[name="rt_msg_source"]:checked').value;
            var content = document.getElementById('rt_msg_content').value || '';

            if (sourceType === 'manual' || !content) {
                resolvedContentCache = '';
                renderPreviewWithContent('');
                return;
            }

            if (content === lastResolvePath && sourceType === lastResolveType && resolvedContentCache) {
                renderPreviewWithContent(resolvedContentCache);
                return;
            }

            lastResolvePath = content;
            lastResolveType = sourceType;

            var previewEl = document.getElementById('rt_msg_preview');
            previewEl.innerHTML = '<span class="text-gray-500">Loading...</span>';

            fetch('/resolve-content', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_type: sourceType, content: content })
            })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.ok) {
                    resolvedContentCache = data.resolved || '';
                    renderPreviewWithContent(resolvedContentCache);
                } else {
                    resolvedContentCache = '';
                    previewEl.innerHTML = '<span class="text-red-400">Error: ' + escapeHtml(data.error || 'Failed') + '</span>';
                }
            })
            .catch(function(err) {
                resolvedContentCache = '';
                previewEl.innerHTML = '<span class="text-red-400">Error loading content</span>';
            });
        }

        function updateMsgPreview() {
            var sourceType = document.querySelector('input[name="rt_msg_source"]:checked').value;

            if (sourceType === 'manual') {
                renderPreviewWithContent(null);
            } else {
                clearTimeout(resolveDebounceTimer);
                resolveDebounceTimer = setTimeout(fetchResolvedContent, 500);
            }
        }

        function renderPreviewWithContent(resolvedContent) {
            var sourceType = document.querySelector('input[name="rt_msg_source"]:checked').value;
            var limit = document.getElementById('rt_mode').value === '2B' ? 32 : 64;
            var preview = '';
            var tag1Start = -1, tag1Len = 0, tag1Type = 0;
            var tag2Start = -1, tag2Len = 0, tag2Type = 0;
            var rtPlusEnabled = document.getElementById('rt_msg_rtplus_enabled').checked;

            if (sourceType === 'manual') {
                if (!rtPlusEnabled) {
                    // Simple text mode
                    preview = document.getElementById('rt_msg_simple_text').value || '';
                    // Update character count
                    var countElem = document.getElementById('rt_msg_simple_count');
                    if (countElem) countElem.textContent = preview.length;
                } else {
                    // RT+ builder mode
                    var prefix = document.getElementById('rt_msg_prefix').value || '';
                    var tag1Text = document.getElementById('rt_msg_tag1_text').value || '';
                    var middle = document.getElementById('rt_msg_middle').value || '';
                    var tag2Text = document.getElementById('rt_msg_tag2_text').value || '';
                    var suffix = document.getElementById('rt_msg_suffix').value || '';
                    tag1Type = parseInt(document.getElementById('rt_msg_tag1_type').value) || 0;
                    tag2Type = parseInt(document.getElementById('rt_msg_tag2_type').value) || 0;

                    preview = prefix + tag1Text;
                    tag1Start = prefix.length;
                    tag1Len = tag1Text.length;

                    if (tag2Text) {
                        preview += middle + tag2Text;
                        tag2Start = prefix.length + tag1Text.length + middle.length;
                        tag2Len = tag2Text.length;
                    }
                    preview += suffix;
                }
            } else {
                var path = document.getElementById('rt_msg_content').value || '';
                var delimiter = document.getElementById('rt_msg_split').value || ' - ';
                tag1Type = parseInt(document.getElementById('rt_msg_tag1_type_auto').value) || 0;
                tag2Type = parseInt(document.getElementById('rt_msg_tag2_type_auto').value) || 0;

                if (!path) {
                    preview = '(enter file path or URL)';
                } else if (resolvedContent) {
                    // Get prefix/suffix
                    var prefix = document.getElementById('rt_msg_prefix_auto').value || '';
                    var suffix = document.getElementById('rt_msg_suffix_auto').value || '';

                    // Calculate RT+ tags BEFORE adding suffix (so suffix isn't included in tag2)
                    if (document.getElementById('rt_msg_rtplus_enabled').checked && delimiter && resolvedContent.indexOf(delimiter) !== -1) {
                        var parts = resolvedContent.split(delimiter, 2);
                        tag1Start = prefix.length;
                        tag1Len = parts[0].length;
                        if (parts.length > 1) {
                            tag2Start = prefix.length + parts[0].length + delimiter.length;
                            tag2Len = parts[1].length; // This is correct - parts[1] doesn't include suffix yet
                        }
                    }

                    // Apply prefix/suffix to preview
                    preview = prefix + resolvedContent + suffix;
                } else {
                    preview = '(no content)';
                }
            }

            // Truncate preview
            if (preview.length > limit) {
                preview = preview.substring(0, limit);
            }

            // Update char count
            var countEl = document.getElementById('rt_msg_char_count');
            var limitEl = document.getElementById('rt_msg_char_limit');
            countEl.textContent = preview.length;
            limitEl.textContent = limit;
            countEl.className = preview.length > limit ? 'text-red-400' : (preview.length > limit - 5 ? 'text-yellow-400' : 'text-green-400');

            // Build preview HTML with highlighted tags (only if RT+ enabled)
            var previewEl = document.getElementById('rt_msg_preview');
            var html = '';
            if (rtPlusEnabled && (tag1Len > 0 || tag2Len > 0)) {
                for (var i = 0; i < preview.length; i++) {
                    var char = escapeHtml(preview[i]);
                    if (i >= tag1Start && i < tag1Start + tag1Len) {
                        html += '<span class="rtplus-tag-1">' + char + '</span>';
                    } else if (i >= tag2Start && i < tag2Start + tag2Len) {
                        html += '<span class="rtplus-tag-2">' + char + '</span>';
                    } else {
                        html += char;
                    }
                }
            } else {
                // No highlighting when RT+ disabled or no tags
                html = escapeHtml(preview);
            }
            previewEl.innerHTML = html || '<span class="text-gray-600">(empty)</span>';

            // Update tag info (hide if RT+ disabled in manual mode)
            var tag1Info = document.getElementById('rt_msg_tag1_info');
            var tag2Info = document.getElementById('rt_msg_tag2_info');
            var tagInfoContainer = document.getElementById('rt_msg_tag_info');

            if (sourceType === 'manual' && !rtPlusEnabled) {
                // Hide tag info in manual mode when RT+ is disabled
                if (tagInfoContainer) tagInfoContainer.style.display = 'none';
            } else {
                if (tagInfoContainer) tagInfoContainer.style.display = 'grid';
                if (tag1Len > 0) {
                    var t1Name = RTPLUS_TYPES[tag1Type] ? RTPLUS_TYPES[tag1Type][0] : 'Unknown';
                    tag1Info.textContent = t1Name + ' [' + tag1Start + '-' + (tag1Start + tag1Len - 1) + ']';
                } else {
                    tag1Info.textContent = '-';
                }
                if (tag2Len > 0) {
                    var t2Name = RTPLUS_TYPES[tag2Type] ? RTPLUS_TYPES[tag2Type][0] : 'Unknown';
                    tag2Info.textContent = t2Name + ' [' + tag2Start + '-' + (tag2Start + tag2Len - 1) + ']';
                } else {
                    tag2Info.textContent = '-';
                }
            }
        }

        function saveRTMessage() {
            var id = document.getElementById('rt_msg_edit_id').value;
            var msg = rtMessages.find(function(m) { return m.id === id; });

            // If msg not found, check if it's a pending new message
            if (!msg && pendingNewMessage && pendingNewMessage.id === id) {
                msg = pendingNewMessage;
                rtMessages.push(msg); // Add to array on save
                pendingNewMessage = null;
            }

            if (!msg) return;

            msg.buffer = currentMsgBuffer;
            msg.cycles = parseInt(document.getElementById('rt_msg_cycles').value) || 2;
            msg.source_type = document.querySelector('input[name="rt_msg_source"]:checked').value;

            msg.rt_plus_enabled = document.getElementById('rt_msg_rtplus_enabled').checked;

            if (msg.source_type === 'manual') {
                if (!msg.rt_plus_enabled) {
                    // Simple text mode
                    msg.content = document.getElementById('rt_msg_simple_text').value;
                    msg.prefix = '';
                    msg.tag1_text = '';
                    msg.middle = '';
                    msg.tag2_text = '';
                    msg.suffix = '';
                    msg.split_delimiter = '';
                } else {
                    // RT+ builder mode
                    msg.prefix = document.getElementById('rt_msg_prefix').value;
                    msg.tag1_text = document.getElementById('rt_msg_tag1_text').value;
                    msg.middle = document.getElementById('rt_msg_middle').value;
                    msg.tag2_text = document.getElementById('rt_msg_tag2_text').value;
                    msg.suffix = document.getElementById('rt_msg_suffix').value;
                    msg.rt_plus_tags = {
                        tag1_type: parseInt(document.getElementById('rt_msg_tag1_type').value),
                        tag2_type: parseInt(document.getElementById('rt_msg_tag2_type').value)
                    };
                    // Build content from parts for backend
                    msg.content = (msg.prefix || '') + (msg.tag1_text || '') +
                        (msg.tag2_text ? (msg.middle || '') + msg.tag2_text : '') + (msg.suffix || '');
                    msg.split_delimiter = msg.middle || ' - ';
                }
            } else {
                // Save file/URL fields
                msg.content = document.getElementById('rt_msg_content').value;
                msg.prefix = document.getElementById('rt_msg_prefix_auto').value;
                msg.suffix = document.getElementById('rt_msg_suffix_auto').value;
                msg.split_delimiter = document.getElementById('rt_msg_split').value;
                msg.rt_plus_enabled = document.getElementById('rt_msg_rtplus_enabled').checked;
                msg.rt_plus_tags = {
                    tag1_type: parseInt(document.getElementById('rt_msg_tag1_type_auto').value),
                    tag2_type: parseInt(document.getElementById('rt_msg_tag2_type_auto').value)
                };
                // Clear manual-only fields
                msg.tag1_text = '';
                msg.middle = '';
                msg.tag2_text = '';
            }

            closeRTMsgModal();
            renderRTMessages();
            syncRTMessages();
        }

        function toggleRTMessage(id) {
            var msg = rtMessages.find(function(m) { return m.id === id; });
            if (msg) {
                msg.enabled = !msg.enabled;
                renderRTMessages();
                syncRTMessages();
            }
        }

        function deleteRTMessage(id) {
            if (!confirm('Delete this message?')) return;
            rtMessages = rtMessages.filter(function(m) { return m.id !== id; });
            renderRTMessages();
            syncRTMessages();
        }

        function syncRTMessages() {
            socket.emit('update', { rt_messages: JSON.stringify(rtMessages) });
        }

        function initRTMsgTagSelects() {
            // Build options HTML
            var opts = '';
            for (var i = 0; i <= 63; i++) {
                var info = RTPLUS_TYPES[i] || ["Unknown", ""];
                opts += '<option value="' + i + '">' + i + ': ' + info[0] + '</option>';
            }

            // Manual mode selects
            var sel1 = document.getElementById('rt_msg_tag1_type');
            var sel2 = document.getElementById('rt_msg_tag2_type');
            if (sel1) { sel1.innerHTML = opts; sel1.value = '4'; }
            if (sel2) { sel2.innerHTML = opts; sel2.value = '1'; }

            // Auto mode selects (file/URL)
            var sel1Auto = document.getElementById('rt_msg_tag1_type_auto');
            var sel2Auto = document.getElementById('rt_msg_tag2_type_auto');
            if (sel1Auto) { sel1Auto.innerHTML = opts; sel1Auto.value = '4'; }
            if (sel2Auto) { sel2Auto.innerHTML = opts; sel2Auto.value = '1'; }
        }

        function initRTPlusBuilder() {
            var select1 = document.getElementById('builder_tag1_type');
            var select2 = document.getElementById('builder_tag2_type');
            if (!select1 || !select2) return;

            select1.innerHTML = '';
            select2.innerHTML = '<option value="-1">None (single tag)</option>';

            for (var i = 0; i <= 63; i++) {
                var info = RTPLUS_TYPES[i] || ["Unknown", ""];
                var opt = '<option value="' + i + '">' + i + ': ' + info[0] + '</option>';
                select1.innerHTML += opt;
                select2.innerHTML += opt;
            }

            select1.value = "4";
            select2.value = "1";
        }

        function openRTPlusModal() {
            initRTPlusBuilder();
            loadBuilderFromState();
            document.getElementById('rtplus_modal').style.display = 'flex';

            // Show/hide buffer tabs based on manual buffer mode
            var manualBuffers = document.getElementById('rt_manual_buffers');
            var bufferTabs = document.getElementById('builder_buffer_tabs');
            if (bufferTabs) {
                bufferTabs.style.display = (manualBuffers && manualBuffers.checked) ? 'flex' : 'none';
            }

            // Pre-populate split source from current RT text
            var rtText = document.getElementById('rt_text');
            if (rtText && rtText.value) {
                document.getElementById('split_source').value = rtText.value;
            }

            // Set mode radio based on current state
            var mode = '{{ state.rt_plus_mode }}' || 'format';
            var radios = document.querySelectorAll('input[name="rtplus_mode"]');
            radios.forEach(function(r) { r.checked = (r.value === mode); });
            updateBuilderPreview();
        }

        function closeRTPlusModal() {
            document.getElementById('rtplus_modal').style.display = 'none';
        }

        function setBuilderBuffer(buf) {
            saveBuilderToLocalState();
            currentBuilderBuffer = buf;

            document.getElementById('builder_tab_a').className = buf === 'a'
                ? 'px-4 py-2 rounded font-bold bg-[#d946ef] text-white'
                : 'px-4 py-2 rounded font-bold bg-[#333] text-gray-400 hover:bg-[#444]';
            document.getElementById('builder_tab_b').className = buf === 'b'
                ? 'px-4 py-2 rounded font-bold bg-[#d946ef] text-white'
                : 'px-4 py-2 rounded font-bold bg-[#333] text-gray-400 hover:bg-[#444]';

            loadBuilderBufferToUI();
            updateBuilderPreview();
        }

        function saveBuilderToLocalState() {
            var s = builderState[currentBuilderBuffer];
            s.prefix = document.getElementById('builder_prefix').value;
            s.tag1_type = parseInt(document.getElementById('builder_tag1_type').value);
            s.tag1_text = document.getElementById('builder_tag1_text').value;
            s.middle = document.getElementById('builder_middle').value;
            s.tag2_type = parseInt(document.getElementById('builder_tag2_type').value);
            s.tag2_text = document.getElementById('builder_tag2_text').value;
            s.suffix = document.getElementById('builder_suffix').value;
        }

        function loadBuilderBufferToUI() {
            var s = builderState[currentBuilderBuffer];
            document.getElementById('builder_prefix').value = s.prefix || '';
            document.getElementById('builder_tag1_type').value = s.tag1_type >= 0 ? s.tag1_type : 4;
            document.getElementById('builder_tag1_text').value = s.tag1_text || '';
            document.getElementById('builder_middle').value = s.middle || ' - ';
            document.getElementById('builder_tag2_type').value = s.tag2_type >= 0 ? s.tag2_type : 1;
            document.getElementById('builder_tag2_text').value = s.tag2_text || '';
            document.getElementById('builder_suffix').value = s.suffix || '';
        }

        function escapeHtml(text) {
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function updateBuilderPreview() {
            var prefix = document.getElementById('builder_prefix').value;
            var tag1Type = parseInt(document.getElementById('builder_tag1_type').value);
            var tag1Text = document.getElementById('builder_tag1_text').value;
            var middle = document.getElementById('builder_middle').value;
            var tag2Type = parseInt(document.getElementById('builder_tag2_type').value);
            var tag2Text = document.getElementById('builder_tag2_text').value;
            var suffix = document.getElementById('builder_suffix').value;

            var tag1Start = prefix.length;
            var tag1End = tag1Start + tag1Text.length;
            var tag2Start = -1, tag2End = -1;

            if (tag2Type >= 0 && tag2Text) {
                tag2Start = tag1End + middle.length;
                tag2End = tag2Start + tag2Text.length;
            }

            // Build preview with highlights
            var html = escapeHtml(prefix);
            if (tag1Text) {
                html += '<span class="rtplus-tag-1">' + escapeHtml(tag1Text) + '</span>';
            }
            if (tag2Type >= 0 && tag2Text) {
                html += escapeHtml(middle);
                html += '<span class="rtplus-tag-2">' + escapeHtml(tag2Text) + '</span>';
            }
            html += escapeHtml(suffix);

            var totalLen = prefix.length + tag1Text.length +
                (tag2Type >= 0 && tag2Text ? middle.length + tag2Text.length : 0) + suffix.length;

            var rtModeEl = document.getElementById('rt_mode');
            var limit = (rtModeEl && rtModeEl.value === '2B') ? 32 : 64;

            document.getElementById('builder_preview').innerHTML = html || '<span class="text-gray-600">Empty message</span>';

            var countEl = document.getElementById('builder_char_count');
            countEl.innerText = totalLen;
            countEl.className = totalLen > limit ? 'text-red-400' : 'text-green-400';
            document.getElementById('builder_char_limit').innerText = limit;

            // Tag info
            document.getElementById('builder_tag1_info').innerText = tag1Text
                ? 'pos ' + tag1Start + ', len ' + tag1Text.length
                : 'not set';
            document.getElementById('builder_tag1_typename').innerText = tag1Type >= 0
                ? RTPLUS_TYPES[tag1Type][0]
                : 'None';

            document.getElementById('builder_tag2_info').innerText = (tag2Type >= 0 && tag2Text)
                ? 'pos ' + tag2Start + ', len ' + tag2Text.length
                : 'not set';
            document.getElementById('builder_tag2_typename').innerText = tag2Type >= 0
                ? RTPLUS_TYPES[tag2Type][0]
                : 'None';
        }

        function loadBuilderFromState() {
            // Load saved builder state from server
            try {
                var savedA = '{{ state.rt_plus_builder_a|e }}';
                var savedB = '{{ state.rt_plus_builder_b|e }}';
                if (savedA) builderState.a = JSON.parse(savedA);
                if (savedB) builderState.b = JSON.parse(savedB);
            } catch (e) {}
            loadBuilderBufferToUI();
        }

        function applyBuilderConfig() {
            saveBuilderToLocalState();

            // Build RT messages from builder state
            function buildMsg(s) {
                var msg = s.prefix + s.tag1_text;
                if (s.tag2_type >= 0 && s.tag2_text) {
                    msg += s.middle + s.tag2_text;
                }
                msg += s.suffix;
                return msg;
            }

            var rtA = buildMsg(builderState.a);
            var rtB = buildMsg(builderState.b);

            // Update RT fields in UI
            var manualBuffers = document.getElementById('rt_manual_buffers');
            if (manualBuffers && manualBuffers.checked) {
                document.getElementById('rt_a').value = rtA;
                document.getElementById('rt_b').value = rtB;
            } else {
                document.getElementById('rt_text').value = rtA;
            }

            // Send builder state to backend
            socket.emit('update', {
                rt_plus_mode: 'builder',
                rt_plus_builder_a: JSON.stringify(builderState.a),
                rt_plus_builder_b: JSON.stringify(builderState.b)
            });

            // Hide legacy format fields since we're using builder mode
            var formatSingle = document.getElementById('rt_format_single');
            var formatDual = document.getElementById('rt_format_dual');
            if (formatSingle) formatSingle.style.display = 'none';
            if (formatDual) formatDual.style.display = 'none';

            sync();
            closeRTPlusModal();
        }

        function resetBuilder() {
            builderState[currentBuilderBuffer] = {
                prefix: '', tag1_type: 4, tag1_text: '',
                middle: ' - ', tag2_type: 1, tag2_text: '', suffix: ''
            };
            loadBuilderBufferToUI();
            updateBuilderPreview();
        }

        function updateRTPlusMode() {
            var mode = document.querySelector('input[name="rtplus_mode"]:checked').value;
            socket.emit('update', { rt_plus_mode: mode });

            // Show/hide legacy format fields based on mode
            var formatSingle = document.getElementById('rt_format_single');
            var formatDual = document.getElementById('rt_format_dual');
            if (formatSingle) formatSingle.style.display = (mode === 'builder') ? 'none' : 'block';
            if (formatDual) formatDual.style.display = (mode === 'builder') ? 'none' : 'flex';
        }

        function showSplitPanel() {
            document.getElementById('split_panel').classList.remove('hidden');
            document.getElementById('btn_show_split').classList.add('hidden');
        }

        function hideSplitPanel() {
            document.getElementById('split_panel').classList.add('hidden');
            document.getElementById('btn_show_split').classList.remove('hidden');
        }

        function performSplit() {
            var source = document.getElementById('split_source').value;
            var delimiter = document.getElementById('split_delimiter').value;

            if (!source || !delimiter) {
                alert('Please enter source text and delimiter');
                return;
            }

            var parts = source.split(delimiter);
            if (parts.length >= 2) {
                // First part goes to Tag 1, second to Tag 2
                document.getElementById('builder_tag1_text').value = parts[0].trim();
                document.getElementById('builder_tag2_text').value = parts[1].trim();
                document.getElementById('builder_middle').value = delimiter;
                document.getElementById('builder_prefix').value = '';
                // If there's a third part, put it in suffix
                if (parts.length > 2) {
                    document.getElementById('builder_suffix').value = delimiter + parts.slice(2).join(delimiter);
                } else {
                    document.getElementById('builder_suffix').value = '';
                }
            } else {
                // Only one part - put it all in Tag 1
                document.getElementById('builder_tag1_text').value = source.trim();
                document.getElementById('builder_tag2_text').value = '';
                document.getElementById('builder_tag2_type').value = '-1';
            }

            hideSplitPanel();
            updateBuilderPreview();
        }

        function setTab(id, evt) {
            document.querySelectorAll('.content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            const tgt = evt ? evt.target : (typeof event !== 'undefined' ? event.target : null);
            if (tgt) tgt.classList.add('active');
        }

        function updatePwr() {
            let btn = document.getElementById('pwrBtn');
            if(running) {
                btn.innerText = "ON AIR";
                btn.classList.add('on');
            } else {
                btn.innerText = "OFF AIR";
                btn.classList.remove('on');
            }
        }
        updatePwr();
        
        socket.on('monitor', function(data) {
            // Heartbeat animation
            let hb = document.getElementById('heartbeat');
            if (hb) hb.style.opacity = (hb.style.opacity == 0.2 ? 1 : 0.2);

            const setText = (id, v) => { const el = document.getElementById(id); if (el) el.innerText = v; };
            setText('live_ps', data.ps);
            setText('live_rt', data.rt);
            setText('live_lps', data.lps);
            setText('live_ptyn', data.ptyn);
            setText('live_af', data.af);
            setText('live_rt_plus', data.rt_plus_info);
            setText('live_pi', data.pi); // Show current PI
            setText('live_pty', pty_list[data.pty_idx] || "None");
            if (typeof data.pilot_generated !== 'undefined') setText('pilot_status', data.pilot_generated ? 'Generated' : 'Disabled (Pass-through/Genlock)');
        });

        // Legacy functions - kept for backward compatibility but now no-ops
        function updateRTVisibility() {
            // Old RT visibility toggle removed - using new message list UI
        }

        function updateCycleControls() {
            // Old cycle controls removed - using new message list UI
        }

        async function saveSettings() {
            const statusEl = document.getElementById('settings_status');
            if (statusEl) statusEl.innerText = 'Saving...';
            const payload = {
                auto_start: document.getElementById('auto_start').checked,
                user: document.getElementById('auth_user').value.trim()
            };
            const passEl = document.getElementById('auth_pass');
            if (passEl && passEl.value.trim()) payload.password = passEl.value.trim();
            try {
                const res = await fetch('/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (res.ok) {
                    if (statusEl) statusEl.innerText = 'Saved. Password updated if provided.';
                    if (passEl) passEl.value = '';
                } else {
                    if (statusEl) statusEl.innerText = 'Save failed (unauthorized or server error).';
                }
            } catch (e) {
                if (statusEl) statusEl.innerText = 'Save failed (network error).';
            }
        }

        function updatePTYList() {
            var isRBDS = document.getElementById('rbds').checked;
            var ptySelect = document.getElementById('pty');
            var currentValue = ptySelect.value;

            // Update option text based on RDS/RBDS
            Array.from(ptySelect.options).forEach(function(option) {
                var rdsText = option.getAttribute('data-rds');
                var rbdsText = option.getAttribute('data-rbds');
                option.textContent = isRBDS ? rbdsText : rdsText;
            });

            // Keep same value selected
            ptySelect.value = currentValue;

            // Sync the rbds checkbox state
            sync();
        }

        function sync() {
            const getVal = (id) => {
                let el = document.getElementById(id);
                if(!el) return null;
                if(el.type === 'checkbox') return el.checked;
                if(el.type === 'number' || el.type === 'range') return parseFloat(el.value);
                return el.value;
            };

            // Safely update any display fields only if the source exists
            const setValText = (valId, srcId) => {
                const valEl = document.getElementById(valId);
                const srcEl = document.getElementById(srcId);
                if (valEl && srcEl) valEl.innerText = srcEl.value;
            };
            setValText('val_rds', 'rds_level');
            setValText('val_pilot', 'pilot_level');
            const genEl = document.getElementById('genlock_offset');
            const valOffset = document.getElementById('val_offset');
            if (genEl && valOffset) valOffset.innerText = genEl.value;

            let data = {
                device_in_idx: getVal('dev_in'),
                device_out_idx: getVal('dev_out'),
                passthrough: getVal('passthrough'),
                genlock: getVal('genlock'),
                genlock_offset: getVal('genlock_offset'),
                rds_level: getVal('rds_level'),
                pilot_level: getVal('pilot_level'),
                pi: getVal('pi'), pty: getVal('pty'), rbds: getVal('rbds'), tp: getVal('tp'), ta: getVal('ta'), ms: getVal('ms'),
                di_stereo: getVal('di_stereo'), di_head: getVal('di_head'), di_comp: getVal('di_comp'), di_dyn: getVal('di_dyn'),
                en_af: getVal('en_af'), af_list: getVal('af_list'), af_method: getVal('af_method'),
                ps_dynamic: getVal('ps_dynamic'), ps_centered: getVal('ps_centered'),
                rt_text: getVal('rt_text'), rt_manual_buffers: getVal('rt_manual_buffers'), rt_cycle_ab: getVal('rt_cycle_ab'),
                rt_a: getVal('rt_a'), rt_b: getVal('rt_b'), rt_mode: getVal('rt_mode'),
                rt_cycle: getVal('rt_cycle'), rt_centered: getVal('rt_centered'), rt_cr: getVal('rt_cr'),
                rt_cycle_time: getVal('rt_cycle_time'),
                rt_ab_cycle_count: getVal('rt_ab_cycle_count'),
                
                // New RT+
                rt_plus_format_a: getVal('rt_plus_format_a'),
                rt_plus_format_b: getVal('rt_plus_format_b'),
                en_rt_plus: getVal('en_rt_plus'),
                
                ptyn: getVal('ptyn'), en_ptyn: getVal('en_ptyn'), ptyn_centered: getVal('ptyn_centered'),
                ecc: getVal('ecc'), lic: getVal('lic'), tz_offset: getVal('tz_offset'), en_ct: getVal('en_ct'), en_id: getVal('en_id'),
                ps_long_32: getVal('ps_long_32'), en_lps: getVal('en_lps'), lps_centered: getVal('lps_centered'), lps_cr: getVal('lps_cr'),
                en_dab: getVal('en_dab'), dab_channel: getVal('dab_channel'),
                dab_eid: getVal('dab_eid'), dab_mode: getVal('dab_mode'), dab_es_flag: getVal('dab_es_flag'),
                dab_sid: getVal('dab_sid'), dab_variant: getVal('dab_variant'),
                en_eon: getVal('en_eon'), eon_services: getVal('eon_services'),
                rds_freq: getVal('rds_freq'),
                group_sequence: getVal('group_sequence'), scheduler_auto: getVal('scheduler_auto')
            };
            socket.emit('update', data);
        }

        // === DATASET FUNCTIONS ===
        var datasets = {};
        var currentDataset = 1;

        function loadDatasets() {
            fetch('/datasets')
                .then(function(res) { return res.json(); })
                .then(function(data) {
                    datasets = data.datasets;
                    currentDataset = data.current;
                    renderDatasetButtons();
                })
                .catch(function(e) { console.error('Failed to load datasets:', e); });
        }

        function renderDatasetButtons() {
            var container = document.getElementById('dataset_buttons');
            if (!container) return;
            container.innerHTML = '';

            for (var num in datasets) {
                var dataset = datasets[num];
                var btn = document.createElement('button');
                btn.className = parseInt(num) === currentDataset ? 'px-4 py-3 rounded text-sm font-semibold bg-pink-600 text-white' : 'px-4 py-3 rounded text-sm font-semibold bg-gray-700 hover:bg-gray-600 text-gray-200';
                btn.textContent = dataset.name || 'Dataset ' + num;
                btn.onclick = (function(n) { return function() { switchDataset(n); }; })(parseInt(num));
                container.appendChild(btn);
            }

            var nameEl = document.getElementById('current_dataset_name');
            if (nameEl && datasets[currentDataset]) {
                nameEl.textContent = datasets[currentDataset].name || 'Dataset ' + currentDataset;
            }
        }

        function switchDataset(num) {
            fetch('/datasets/' + num + '/switch', { method: 'POST' })
                .then(function(res) {
                    if (res.ok) {
                        location.reload();
                    }
                })
                .catch(function(e) { alert('Failed to switch dataset'); });
        }

        function createDataset() {
            var nextNum = Math.max.apply(Math, Object.keys(datasets).map(function(n) { return parseInt(n); })) + 1;
            var name = prompt('Enter dataset name:', 'Dataset ' + nextNum);
            if (!name) return;

            fetch('/datasets/' + nextNum, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, state: {} })
            })
                .then(function(res) {
                    if (res.ok) {
                        loadDatasets();
                    }
                })
                .catch(function(e) { alert('Failed to create dataset'); });
        }

        function renameCurrentDataset() {
            var currentName = datasets[currentDataset] ? datasets[currentDataset].name : 'Dataset ' + currentDataset;
            var newName = prompt('Enter new name:', currentName);
            if (!newName) return;

            datasets[currentDataset].name = newName;
            fetch('/datasets/' + currentDataset, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(datasets[currentDataset])
            })
                .then(function(res) {
                    if (res.ok) {
                        loadDatasets();
                    }
                })
                .catch(function(e) { alert('Failed to rename dataset'); });
        }

        function deleteCurrentDataset() {
            if (Object.keys(datasets).length <= 1) {
                alert('Cannot delete the last dataset');
                return;
            }
            var currentName = datasets[currentDataset] ? datasets[currentDataset].name : 'Dataset ' + currentDataset;
            if (!confirm('Delete dataset "' + currentName + '"?')) return;

            fetch('/datasets/' + currentDataset, { method: 'DELETE' })
                .then(function(res) {
                    if (res.ok) {
                        location.reload();
                    }
                })
                .catch(function(e) { alert('Failed to delete dataset'); });
        }

        // === EON FUNCTIONS ===
        var eonServices = [];

        function loadEONServices() {
            var getVal = function(id) {
                var el = document.getElementById(id);
                if (!el) return null;
                if (el.type === 'checkbox') return el.checked;
                return el.value;
            };
            try {
                eonServices = JSON.parse(getVal('eon_services') || '[]');
            } catch (e) {
                eonServices = [];
            }
            updateEONDisplay();
        }

        function updateEONDisplay() {
            var display = document.getElementById('eon_services_display');
            if (!display) return;

            if (eonServices.length === 0) {
                display.innerHTML = '<div class="text-xs text-gray-400">No services configured</div>';
            } else {
                var html = '<div class="text-xs space-y-1">';
                for (var i = 0; i < eonServices.length; i++) {
                    var svc = eonServices[i];
                    html += '<div class="p-2 bg-gray-800 hover:bg-gray-700 rounded cursor-pointer border border-gray-700 hover:border-pink-600 transition-colors" onclick="openEONModalAndEdit(' + i + ')">';
                    html += '<span class="font-mono text-pink-400">' + (svc.pi_on || 'C000') + '</span> - <span class="text-gray-200">' + (svc.ps || 'UNKNOWN') + '</span>';
                    if (svc.af_list) html += ' <span class="text-xs text-gray-500">(' + svc.af_list + ')</span>';
                    html += '</div>';
                }
                html += '</div>';
                display.innerHTML = html;
            }
        }

        function openEONModal() {
            loadEONServices();
            renderEONServiceList();
            document.getElementById('eon_modal').style.display = 'flex';
            document.getElementById('eon_edit_form').style.display = 'none';
        }

        function closeEONModal() {
            document.getElementById('eon_modal').style.display = 'none';
            document.getElementById('eon_edit_form').style.display = 'none';
        }

        function renderEONServiceList() {
            var container = document.getElementById('eon_service_list');
            container.innerHTML = '';

            if (eonServices.length === 0) {
                container.innerHTML = '<div class="text-xs text-gray-400 text-center py-4">No services configured. Click Add Service to create one.</div>';
                return;
            }

            for (var i = 0; i < eonServices.length; i++) {
                var svc = eonServices[i];
                var card = document.createElement('div');
                card.className = 'bg-black border border-gray-700 rounded p-3 flex justify-between items-center';

                var info = document.createElement('div');
                info.className = 'flex-1';
                info.innerHTML = '<div class="font-mono text-sm text-pink-400">' + (svc.pi_on || 'C000') + '</div>' +
                                '<div class="text-sm">' + (svc.ps || 'UNKNOWN') + '</div>' +
                                '<div class="text-xs text-gray-400">' + (svc.af_list || 'No AFs') + '</div>';

                var actions = document.createElement('div');
                actions.className = 'flex gap-2';

                var editBtn = document.createElement('button');
                editBtn.textContent = 'Edit';
                editBtn.className = 'px-3 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs';
                editBtn.onclick = (function(idx) { return function() { editEONService(idx); }; })(i);

                var delBtn = document.createElement('button');
                delBtn.textContent = 'Delete';
                delBtn.className = 'px-3 py-1 bg-red-600 hover:bg-red-500 rounded text-xs';
                delBtn.onclick = (function(idx) { return function() { deleteEONService(idx); }; })(i);

                actions.appendChild(editBtn);
                actions.appendChild(delBtn);
                card.appendChild(info);
                card.appendChild(actions);
                container.appendChild(card);
            }
        }

        function addEONService() {
            document.getElementById('eon_edit_idx').value = '';
            document.getElementById('eon_pi').value = '';
            document.getElementById('eon_ps').value = '';
            document.getElementById('eon_pty').value = '0';
            document.getElementById('eon_af').value = '';
            document.getElementById('eon_tp').checked = false;
            document.getElementById('eon_ta').checked = false;
            document.getElementById('eon_modal_title').textContent = 'Add EON Service';
            document.getElementById('eon_edit_form').style.display = 'block';
        }

        function editEONService(idx) {
            var svc = eonServices[idx];
            document.getElementById('eon_edit_idx').value = idx;
            document.getElementById('eon_pi').value = svc.pi_on || '';
            document.getElementById('eon_ps').value = svc.ps || '';
            document.getElementById('eon_pty').value = svc.pty || 0;
            document.getElementById('eon_af').value = svc.af_list || '';
            document.getElementById('eon_tp').checked = svc.tp || false;
            document.getElementById('eon_ta').checked = svc.ta || false;
            document.getElementById('eon_modal_title').textContent = 'Edit EON Service';
            document.getElementById('eon_edit_form').style.display = 'block';
        }

        function deleteEONService(idx) {
            if (!confirm('Delete this EON service?')) return;
            eonServices.splice(idx, 1);
            syncEONServices();
            renderEONServiceList();
            updateEONDisplay();
        }

        function cancelEONEdit() {
            document.getElementById('eon_edit_form').style.display = 'none';
            document.getElementById('eon_modal_title').textContent = 'Manage EON Services';
        }

        function saveEONService() {
            var idx = document.getElementById('eon_edit_idx').value;
            var svc = {
                pi_on: document.getElementById('eon_pi').value.toUpperCase() || 'C000',
                ps: document.getElementById('eon_ps').value || 'UNKNOWN',
                pty: parseInt(document.getElementById('eon_pty').value) || 0,
                af_list: document.getElementById('eon_af').value || '',
                tp: document.getElementById('eon_tp').checked ? 1 : 0,
                ta: document.getElementById('eon_ta').checked ? 1 : 0
            };

            if (idx === '') {
                eonServices.push(svc);
            } else {
                eonServices[parseInt(idx)] = svc;
            }

            syncEONServices();
            renderEONServiceList();
            updateEONDisplay();
            cancelEONEdit();
        }

        function syncEONServices() {
            var hiddenInput = document.getElementById('eon_services');
            if (hiddenInput) {
                hiddenInput.value = JSON.stringify(eonServices);
            }
            socket.emit('update', { eon_services: JSON.stringify(eonServices) });
        }

        function openEONModalAndEdit(idx) {
            openEONModal();
            setTimeout(function() {
                editEONService(idx);
            }, 100);
        }

        function togglePower() {
            running = !running;
            updatePwr();
            if(running) {
                socket.emit('control', { 
                    action: 'start', 
                    dev_in: document.getElementById('dev_in').value, 
                    dev_out: document.getElementById('dev_out').value 
                });
            } else {
                socket.emit('control', { action: 'stop' });
            }
        }
        
        // Initialize cycle controls on load
        updateCycleControls();

        setInterval(() => {
             let hasInput = document.getElementById('dev_in').value != "-1";
             let gl = document.getElementById('genlock');
             if(!hasInput) {
                 if(!gl.disabled) { gl.checked = false; gl.disabled = true; gl.parentElement.classList.add('opacity-50'); sync(); }
             } else {
                 if(gl.disabled) { gl.disabled = false; gl.parentElement.classList.remove('opacity-50'); }
             }

             // Disable pilot slider when pass-through is active and an input device is present
             const pt = document.getElementById('passthrough');
             const pilotSlider = document.getElementById('pilot_level');
             const pilotStatus = document.getElementById('pilot_status');
             if (pilotSlider && pt) {
                 const pilotDisabled = hasInput && pt.checked;
                 if (pilotDisabled) {
                     if (!pilotSlider.disabled) { pilotSlider.disabled = true; pilotSlider.parentElement.classList.add('opacity-50'); if (pilotStatus) pilotStatus.innerText = 'Disabled (Pass-through)'; }
                 } else {
                     if (pilotSlider.disabled) { pilotSlider.disabled = false; pilotSlider.parentElement.classList.remove('opacity-50'); if (pilotStatus) pilotStatus.innerText = 'Enabled'; }
                 }
             }

               // Update cycle controls visibility regularly
               updateCycleControls();
        }, 1000);

        // Initialize RT Messages on page load
        initRTMsgTagSelects();
        renderRTMessages();
        updatePTYList(); // Initialize PTY list based on RDS/RBDS setting
        updateAFMethodUI(); // Initialize AF UI based on method
        loadDatasets(); // Initialize datasets on page load
        loadEONServices(); // Initialize EON services on page load

        // === AF PAIR FUNCTIONS ===
        var afPairs = [];
        var pendingNewAFPair = null;

        function updateAFMethodUI() {
            var method = document.getElementById('af_method').value;
            var methodAUI = document.getElementById('af_method_a_ui');
            var methodBUI = document.getElementById('af_method_b_ui');

            if (method === 'A') {
                methodAUI.style.display = 'block';
                methodBUI.style.display = 'none';
            } else {
                methodAUI.style.display = 'none';
                methodBUI.style.display = 'block';
                parseAFPairs(); // Load pairs from af_list
            }
        }

        function parseAFPairs() {
            // Load AF pairs from state
            try {
                var pairsJson = '{{ state.af_pairs|safe }}';
                afPairs = JSON.parse(pairsJson) || [];
            } catch(e) {
                // Fallback: parse af_list into a single pair for backward compatibility
                var afList = document.getElementById('af_list').value;
                if (afList) {
                    var freqs = afList.split(',').map(function(f) { return f.trim(); }).filter(function(f) { return f; });
                    if (freqs.length > 0) {
                        afPairs = [{
                            id: 'pair_1',
                            main: freqs[0],
                            alts: freqs.slice(1).join(', ')
                        }];
                    }
                }
            }
            renderAFPairs();
        }

        function renderAFPairs() {
            var container = document.getElementById('af_pairs_list');
            var emptyState = document.getElementById('af_pairs_empty');

            if (!afPairs || afPairs.length === 0) {
                container.innerHTML = '';
                emptyState.style.display = 'block';
                return;
            }

            emptyState.style.display = 'none';
            container.innerHTML = '';

            afPairs.forEach(function(pair, index) {
                var card = document.createElement('div');
                card.className = 'rt-msg-card';
                card.setAttribute('data-id', pair.id);

                var header = document.createElement('div');
                header.className = 'rt-msg-header';
                header.style.cssText = 'display: flex; align-items: center;';

                var editArea = document.createElement('div');
                editArea.style.cssText = 'cursor:pointer; flex: 1; display: flex; align-items: center;';
                editArea.onclick = function() { editAFPair(pair.id); };

                var pairLabel = document.createElement('span');
                pairLabel.className = 'text-sm font-bold text-[#7c3aed]';
                pairLabel.textContent = 'Pair ' + (index + 1) + ': ' + pair.main + ' MHz';

                var arrow = document.createElement('span');
                arrow.className = 'text-xs text-gray-500 mx-2';
                arrow.textContent = '→';

                var altsLabel = document.createElement('span');
                altsLabel.className = 'text-xs text-gray-400';
                altsLabel.textContent = pair.alts || '(no alternates)';

                editArea.appendChild(pairLabel);
                editArea.appendChild(arrow);
                editArea.appendChild(altsLabel);

                if (pair.regional) {
                    var rvBadge = document.createElement('span');
                    rvBadge.className = 'text-[10px] bg-orange-600 text-white px-2 py-0.5 rounded ml-2';
                    rvBadge.textContent = 'RV';
                    editArea.appendChild(rvBadge);
                }

                var deleteBtn = document.createElement('button');
                deleteBtn.className = 'text-sm hover:text-red-400 ml-2';
                deleteBtn.style.cssText = 'flex-shrink: 0;';
                deleteBtn.textContent = '×';
                deleteBtn.onclick = function(e) {
                    e.stopPropagation();
                    deleteAFPair(pair.id);
                };

                header.appendChild(editArea);
                header.appendChild(deleteBtn);
                card.appendChild(header);
                container.appendChild(card);
            });
        }

        function addAFPair() {
            pendingNewAFPair = {
                id: 'pair_' + Date.now(),
                main: '',
                alts: '',
                regional: false
            };
            openAFPairModal(pendingNewAFPair, true);
        }

        function editAFPair(id) {
            var pair = afPairs.find(function(p) { return p.id === id; });
            if (!pair) return;
            pendingNewAFPair = null;
            openAFPairModal(pair, false);
        }

        function openAFPairModal(pair, isNew) {
            document.getElementById('af_pair_edit_id').value = pair.id;
            document.getElementById('af_pair_modal_title').textContent = isNew ? 'Add Frequency Pair' : 'Edit Frequency Pair';
            document.getElementById('af_pair_main').value = pair.main || '';
            document.getElementById('af_pair_alts').value = pair.alts || '';
            document.getElementById('af_pair_regional').checked = pair.regional || false;
            document.getElementById('af_pair_modal').style.display = 'flex';
        }

        function closeAFPairModal() {
            pendingNewAFPair = null;
            document.getElementById('af_pair_modal').style.display = 'none';
        }

        function saveAFPair() {
            var id = document.getElementById('af_pair_edit_id').value;
            var pair = afPairs.find(function(p) { return p.id === id; });

            if (!pair && pendingNewAFPair && pendingNewAFPair.id === id) {
                pair = pendingNewAFPair;
                afPairs.push(pair);
                pendingNewAFPair = null;
            }

            if (!pair) return;

            pair.main = document.getElementById('af_pair_main').value.trim();
            pair.alts = document.getElementById('af_pair_alts').value.trim();
            pair.regional = document.getElementById('af_pair_regional').checked;

            closeAFPairModal();
            renderAFPairs();
            syncAFPairs();
        }

        function deleteAFPair(id) {
            afPairs = afPairs.filter(function(p) { return p.id !== id; });
            renderAFPairs();
            syncAFPairs();
        }

        function syncAFPairs() {
            // Store pairs as JSON in af_pairs field
            socket.emit('update', { af_pairs: JSON.stringify(afPairs) });

            // Build af_list for display - show all unique main frequencies
            if (afPairs.length === 0) {
                document.getElementById('af_list').value = '';
            } else {
                // Collect all unique frequencies from all pairs
                var allFreqs = [];
                var seen = {};

                afPairs.forEach(function(pair) {
                    if (pair.main && !seen[pair.main]) {
                        allFreqs.push(pair.main);
                        seen[pair.main] = true;
                    }
                    if (pair.alts) {
                        var altFreqs = pair.alts.split(',').map(function(f) { return f.trim(); }).filter(function(f) { return f; });
                        altFreqs.forEach(function(alt) {
                            if (!seen[alt]) {
                                allFreqs.push(alt);
                                seen[alt] = true;
                            }
                        });
                    }
                });

                document.getElementById('af_list').value = allFreqs.join(', ');
            }
        }

    </script>
</body>
</html>
"""

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
