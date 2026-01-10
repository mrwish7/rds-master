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
from datetime import datetime, timezone, date
from flask import Flask, render_template_string, request, redirect, url_for, session
from flask_socketio import SocketIO
from scipy import signal as dsp_signal
from scipy.fft import fft

# --- SETTINGS ---
REQUIRE_MME = True 

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


PTY_LIST = ["None", "News", "Current Affairs", "Information", "Sport", "Education", "Drama", "Culture", "Science", "Varied", "Pop Music", "Rock Music", "Easy Music", "Light Classical", "Serious Classical", "Other Music", "Weather", "Finance", "Children's", "Social Affairs", "Religion", "Phone-In", "Travel", "Leisure", "Jazz", "Country", "National Music", "Oldies", "Folk Music", "Documentary", "Alarm Test", "Alarm"]

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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

auth_config = {"user": "admin", "pass": "pass"}

# --- DEFAULT STATE ---
default_state = {
    "running": False,
    # Audio
    "device_out_idx": 0, "device_in_idx": -1,
    "genlock": False, "passthrough": False,
    "pilot_level": 0.0, "rds_level": 4.5, "genlock_offset": 0.0,
    
    # Basic
    "pi": "2FFF", "pty": 0, "tp": 1, "ta": 0, "ms": 1,
    "di_stereo": 1, "di_head": 0, "di_comp": 1, "di_dyn": 0,
    "en_af": 1, "af_list": "87.6, 87.7", "af_method": "A",
    
    # Text
    "ps_dynamic": "RDS PRO", "ps_centered": False,
    "rt_text": "RDS MASTER PRO", "rt_manual_buffers": False, "rt_cycle_ab": False,
    "rt_a": "RDS MASTER PRO", "rt_b": "Simple & Open Source RDS Encoder",
    "rt_cr": True, "rt_centered": False,
    "rt_mode": "2A", "rt_cycle": True, "rt_cycle_time": 5, "rt_active_buffer": 0,
    
    # RT+
    "rt_plus_format_a": "{artist} - {title}", 
    "rt_plus_format_b": "{artist} - {title}",
    "en_rt_plus": False,
    
    # Expert
    "ecc": "E3", "lic": "09", "tz_offset": 0.0, "en_ct": 1, "en_id": 1,
    "ps_long_32": "RDS MASTER PRO v10.14", "en_lps": 1, "lps_centered": False,
    "ptyn": "PYTHON", "en_ptyn": 1, "ptyn_centered": False,
    "en_dab": 0, "dab_channel": "12B", "dab_eid": "CE15", "dab_mode": 1, "dab_es_flag": 0,
    "dab_sid": "0000", "dab_variant": 0,

    # Settings
    "auto_start": True,
    
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
class RTPlusParser:
    @staticmethod
    def parse(text, fmt_str, centered=False, limit=64):
        tags = []
        if not text or not fmt_str: return tags
        
        offset = 0
        if centered and len(text) < limit:
            offset = (limit - len(text)) // 2
            
        pattern = re.escape(fmt_str)
        pattern = pattern.replace(r"\{artist\}", r"(?P<artist>.+)")
        pattern = pattern.replace(r"\{title\}", r"(?P<title>.+)")
        
        match = re.search(pattern, text)
        if match:
            for name, group_idx in match.groupdict().items():
                raw_start = match.start(name)
                length = len(match.group(name))
                
                real_start = raw_start + offset
                # 1=Title, 4=Artist
                c_type = 1 if name == "title" else 4
                
                # Length Marker is (len - 1). 
                # Bounds check: RT max 64.
                if real_start < 64 and length > 0:
                    if real_start + length > 64: length = 64 - real_start
                    tags.append((c_type, real_start, length))
        return tags

# --- DEVICE DISCOVERY ---
def get_valid_devices():
    valid_inputs = []
    valid_outputs = []
    try:
        devs = sd.query_devices()
        apis = sd.query_hostapis()
        for i, d in enumerate(devs):
            api_name = apis[d['hostapi']]['name']
            if REQUIRE_MME and "MME" not in api_name: continue
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

def save_config():
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
    config['RDS'] = {k: str(v) for k, v in state.items()}
    config['AUTH'] = {'user': auth_config.get('user','admin'), 'pass': auth_config.get('pass','admin')}
    with open(CONFIG_FILE, 'w') as f: config.write(f)

def sig_abort(sig, frame):
    state["running"] = False
    save_config()
    os._exit(0)
signal.signal(signal.SIGINT, sig_abort)

# --- WORKERS ---
def parse_text_source(text):
    if not text: return ""
    try:
        if "\\" in text:
            # Track if any substitution failed
            failed = False
            def file_repl(m):
                nonlocal failed
                try: 
                    with open(m.group(1), 'r', encoding='utf-8') as f: return f.read().strip()
                except: 
                    failed = True
                    return ""  # Return empty on error
            def url_repl(m):
                nonlocal failed
                try:
                    with urllib.request.urlopen(m.group(1), timeout=2) as r: return r.read().decode('utf-8').strip()
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
    def to_state(data):
        global state
        changed = False
        for k, v in data.items():
            if k in state:
                try:
                    if isinstance(state[k], bool): state[k] = bool(v)
                    elif isinstance(state[k], float): state[k] = float(v)
                    elif isinstance(state[k], int): state[k] = int(v)
                    else: state[k] = v
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

    def get_text(self, key):
        val = state.get(key, "")
        return resolved_cache.get(key, val) if "\\" in val else val

    def freq_code(self, f):
        try: return round((float(f) - 87.5) / 0.1) if 87.6 <= float(f) <= 107.9 else 205
        except: return 205

    def split(self, text, width=8, center=False):
        words, frames, curr = text.split(), [], ""
        pad = lambda s: s.center(width) if center else s.ljust(width)
        for w in words:
            if len(w) > width:
                if curr: frames.append(pad(curr)); curr = ""
                chunks = [w[i:i+width] for i in range(0, len(w), width)]
                for c in chunks[:-1]: frames.append(pad(c))
                curr = chunks[-1]
            else:
                test = (curr + " " + w).strip() if curr else w
                if len(test) <= width: curr = test
                else: frames.append(pad(curr)); curr = w
        if curr: frames.append(pad(curr))
        return frames

    def parse_smart(self, raw, width, center):
        seq = []
        if re.match(r"\s*\d+s:", raw):
            # Only treat timed syntax when the string starts with Ns: tokens.
            # Split on whitespace-delimited slashes so literal slashes remain intact.
            for p in re.split(r"\s*/\s*", raw):
                m = re.match(r"(\d+)s:(.*)", p.strip())
                if m:
                    for sf in self.split(m.group(2), width, center): seq.append((int(m.group(1)), sf))
                else:
                    for sf in self.split(p.strip(), width, center): seq.append((2.5, sf))
        else:
            if not raw.strip(): return [(10, " "*width)]
            if len(raw.strip()) <= width: return [(10, self.split(raw, width, center)[0])]
            for sf in self.split(raw.strip(), width, center): seq.append((2.5, sf))
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
        seq = [(0,0), (0,0), (2,0), (0,0)] 
        if state["en_lps"]: seq.append((15,0))
        if state["en_ptyn"]: seq.append((10,0))
        if state["en_id"]: seq.append((1,0))
        if state.get("en_dab"): seq.append((3,0))
        if state["en_rt_plus"]: 
            seq.append((3,0)) 
            seq.append((11,0)) 
        seq.extend([(0,0), (2,0)])
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
                 afs = [x.strip() for x in state["af_list"].split(',') if x.strip()]
                 if afs:
                     if self.af_ptr == 0: b3, self.af_ptr = (224+len(afs))<<8 | self.freq_code(afs[0]), 1
                     else: 
                         f1 = self.freq_code(afs[self.af_ptr])
                         f2 = self.freq_code(afs[self.af_ptr+1]) if self.af_ptr+1 < len(afs) else 205
                         b3, self.af_ptr = (f1<<8)|f2, (self.af_ptr+2) if self.af_ptr+2 < len(afs) else 0
            
            if g_ver == 1: b3 = int(state["pi"], 16)
            return RDSHelper.get_group_bits(0, g_ver, tail, b3, (ord(txt[seg*2])<<8)|ord(txt[seg*2+1]))

        elif g_type == 2:
            # Determine buffer selection
            if state["rt_manual_buffers"]:
                # Manual mode: use traditional cycling
                buf = int((time.time()-self.start_time)/state["rt_cycle_time"])%2 if state["rt_cycle"] else state["rt_active_buffer"]
                raw = self.get_text("rt_a" if buf==0 else "rt_b")
            else:
                # Auto mode: single buffer with sequence support
                raw_input = self.get_text("rt_text")
                limit = 32 if state["rt_mode"] == "2B" else 64
                
                # Check if we need to rebuild the sequence (without centering for now)
                # Compare against the original input, not the first sequence item
                if not self.rt_sequence or raw_input != self.last_rt_text_content:
                    # Parse without centering - we'll apply it later with CR
                    self.rt_sequence = self.parse_smart(raw_input, limit, False)
                    self.rt_seq_idx = 0
                    self.rt_seq_start_time = time.time()
                    self.last_rt_text_content = raw_input
                    if not state["rt_cycle_ab"]:
                        self.rt_ab_flag = 1 - self.rt_ab_flag  # Toggle A/B on content change
                
                dur, txt = self.rt_sequence[self.rt_seq_idx % len(self.rt_sequence)]
                
                # Only advance sequence if there are multiple messages AND time has elapsed
                if len(self.rt_sequence) > 1 and time.time() - self.rt_seq_start_time >= dur:
                    self.rt_seq_idx += 1
                    self.rt_seq_start_time = time.time()
                    dur, txt = self.rt_sequence[self.rt_seq_idx % len(self.rt_sequence)]
                    if not state["rt_cycle_ab"]:
                        self.rt_ab_flag = 1 - self.rt_ab_flag  # Toggle A/B when changing messages
                
                # Handle rt_cycle_ab toggle (cycle same message on A/B)
                if state["rt_cycle_ab"]:
                    buf = int((time.time()-self.start_time)/state["rt_cycle_time"])%2
                else:
                    buf = self.rt_ab_flag
                
                # Use the text from sequence (strip it to get raw content)
                raw = txt.strip()
            
            sig = f"{raw}_{state['rt_centered']}_{state['rt_cr']}"
            limit = 32 if state["rt_mode"] == "2B" else 64
            
            if raw != self.last_rt_clean:
                self.last_rt_clean = raw
                self.rt_plus_toggle = 1 - self.rt_plus_toggle
                fmt = state["rt_plus_format_a"] if buf==0 else state["rt_plus_format_b"]
                self.rt_plus_tags = RTPlusParser.parse(raw, fmt, centered=state['rt_centered'], limit=limit)
                
                tag_str = []
                display_clean = (raw + '\r') if state["rt_cr"] else raw.center(limit) if state["rt_centered"] else raw.ljust(limit)
                for t in self.rt_plus_tags:
                    t_name = "Title" if t[0]==1 else "Artist"
                    content = display_clean[t[1]:t[1]+t[2]]
                    tag_str.append(f"{t_name}: {content}")
                monitor_data["rt_plus_info"] = " | ".join(tag_str)
                
            if sig != self.last_rt_content: self.rt_ptr, self.last_rt_content = 0, sig
            clean = (raw + '\r') if state["rt_cr"] else raw.center(limit) if state["rt_centered"] else raw.ljust(limit)
            
            monitor_data["rt"] = clean

            v = g_ver 
            bpg = 2 if v==1 else 4
            if self.rt_ptr * bpg >= len(clean) or (clean.find('\r') != -1 and self.rt_ptr*bpg > clean.find('\r')): self.rt_ptr = 0
            pad = clean.ljust(64)
            a = self.rt_ptr % 16
            self.rt_ptr += 1
            b3_val = (ord(pad[a*4])<<8)|ord(pad[a*4+1]) if v==0 else 0
            b4_val = (ord(pad[a*4+2])<<8)|ord(pad[a*4+3]) if v==0 else (ord(pad[a*2])<<8)|ord(pad[a*2+1])
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

        elif g_type == 15 and state["en_lps"]:
            raw = self.get_text("ps_long_32")
            if raw != self.last_lps_content: self.last_lps_content, self.lps_ptr = raw, 0
            if not self.lps_sequence or raw != self.lps_sequence[0][1].strip(): 
                self.lps_sequence = self.parse_smart(raw, 32, state['lps_centered'])
            dur, txt = self.lps_sequence[self.lps_seq_idx % len(self.lps_sequence)]
            
            monitor_data["lps"] = txt

            if (time.time() - self.lps_seq_start_time) >= dur:
                self.lps_seq_idx += 1
                self.lps_seq_start_time, self.lps_ptr = time.time(), 0
                dur, txt = self.lps_sequence[self.lps_seq_idx % len(self.lps_sequence)]
            lps_txt = txt.encode('utf-8').ljust(32)[:32]
            seg = self.lps_ptr % 8
            self.lps_ptr += 1
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
        self.taps = dsp_signal.firwin(301, BITRATE * 1.5, fs=SAMPLE_RATE, window=('kaiser', 8.0))
        self.zi = np.zeros(300)

    def process_frame(self, outdata, frames, indata=None):
        lvl_pilot, lvl_rds = (state["pilot_level"]/100.0), (state["rds_level"]/100.0)
        
        use_genlock = state["genlock"] and indata is not None and len(indata) == frames
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
        
        rds_sig = shaped * np.sin(2 * np.pi * RDS_FREQ * t + self.p_rds) * lvl_rds
        pilot_sig = 0.0
        if not use_genlock and not (state.get("passthrough") and indata is not None):
             pilot_sig = np.sin(2 * np.pi * PILOT_FREQ * t + self.p_pilot) * lvl_pilot
             self.p_pilot = (self.p_pilot + 2 * np.pi * PILOT_FREQ * frames / SAMPLE_RATE) % (2 * np.pi)

        self.p_rds = (self.p_rds + 2 * np.pi * RDS_FREQ * frames / SAMPLE_RATE) % (2 * np.pi)
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
    return render_template_string(UI_HTML, inputs=inputs, outputs=outputs, state=state, pty_list=PTY_LIST, auth_config=auth_config)

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

@socketio.on('update')
def handle_update(data): Sanitize.to_state(data)

@socketio.on('control')
def handle_control(data):
    if data['action'] == 'start':
        state["device_out_idx"] = int(data["dev_out"])
        state["device_in_idx"] = int(data["dev_in"])
        state["running"] = True
        save_config()
        threading.Thread(target=run_audio, daemon=True).start()
    else: 
        state["running"] = False
        save_config()

load_config()

def auto_start_if_enabled():
    if state.get("auto_start") and not state.get("running"):
        state["running"] = True
        threading.Thread(target=run_audio, daemon=True).start()

auto_start_if_enabled()

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
                        <div><label>Program Type (PTY)</label><select id="pty" onchange="sync()">{% for p in pty_list %}<option value="{{loop.index0}}" {% if loop.index0 == state.pty %}selected{% endif %}>{{p}}</option>{% endfor %}</select></div>
                    </div>
                </div>
            
                <div class="section">
                    <div class="section-header">RadioText (64-Char)</div>
                    <div class="section-body">
                        <div class="flex justify-between items-center mb-2">
                             <label>Manually specify buffers</label>
                             <input type="checkbox" class="toggle-checkbox" id="rt_manual_buffers" {% if state.rt_manual_buffers %}checked{% endif %} onchange="sync(); setTimeout(updateRTVisibility, 100)">
                        </div>
                        
                        <div id="rt_single_mode" style="display: {% if state.rt_manual_buffers %}none{% else %}block{% endif %}">
                            <div class="mb-1">
                                <label>RadioText</label>
                                <div class="text-[9px] text-gray-500 mb-1">Supports timing: 5s:Message1/10s:Message2. A/B flag toggles on message change.</div>
                                <input type="text" id="rt_text" value="{{state.rt_text}}" onchange="sync()">
                            </div>
                            <div class="mb-2">
                                <label class="text-orange-400">RT+ Format</label>
                                <input type="text" id="rt_plus_format_a" value="{{state.rt_plus_format_a}}" placeholder="{artist} - {title}" onchange="sync()" class="border-orange-900 bg-[#221]">
                            </div>
                            <div class="flex justify-between items-center bg-[#111] p-2 rounded mb-2">
                                <div>
                                    <label>Cycle same message on A/B</label>
                                    <div class="text-[9px] text-gray-500">Toggle A/B at interval (ignores message-based toggle)</div>
                                </div>
                                <input type="checkbox" class="toggle-checkbox" id="rt_cycle_ab" {% if state.rt_cycle_ab %}checked{% endif %} onchange="sync()">
                            </div>
                        </div>
                        
                        <div id="rt_dual_mode" style="display: {% if state.rt_manual_buffers %}block{% else %}none{% endif %}">
                            <div class="flex gap-2 mb-1">
                                 <div class="flex-1"><label>Buffer A</label><input type="text" id="rt_a" value="{{state.rt_a}}" onchange="sync()"></div>
                                 <div class="flex-1"><label>Buffer B</label><input type="text" id="rt_b" value="{{state.rt_b}}" onchange="sync()"></div>
                            </div>
                            <div class="flex gap-2 mb-2">
                                 <div class="flex-1">
                                     <label class="text-orange-400">RT+ Format A</label>
                                     <input type="text" id="rt_plus_format_a" value="{{state.rt_plus_format_a}}" placeholder="{artist} - {title}" onchange="sync()" class="border-orange-900 bg-[#221]">
                                 </div>
                                 <div class="flex-1">
                                     <label class="text-orange-400">RT+ Format B</label>
                                     <input type="text" id="rt_plus_format_b" value="{{state.rt_plus_format_b}}" placeholder="{artist} - {title}" onchange="sync()" class="border-orange-900 bg-[#221]">
                                 </div>
                            </div>
                        </div>
                        
                        <div class="flex justify-between items-center bg-[#111] p-2 rounded">
                             <div class="flex gap-4">
                                 <div><label>Mode</label><select id="rt_mode" onchange="sync()"><option value="2A">2A (64)</option><option value="2B">2B (32)</option></select></div>
                                 <div><label>Time</label><input type="number" id="rt_cycle_time" value="{{state.rt_cycle_time}}" class="w-12" onchange="sync()"></div>
                             </div>
                             <div class="flex gap-4">
                                 <div class="flex flex-col items-center"><label>RT+ Enable</label><input type="checkbox" class="toggle-checkbox" id="en_rt_plus" {% if state.en_rt_plus %}checked{% endif %} onchange="sync()"></div>
                                 <div class="flex flex-col items-center"><label>Centre</label><input type="checkbox" class="toggle-checkbox" id="rt_centered" {% if state.rt_centered %}checked{% endif %} onchange="if(this.checked) document.getElementById('rt_cr').checked = false; sync()"></div>
                                 <div class="flex flex-col items-center"><label>Append CR</label><input type="checkbox" class="toggle-checkbox" id="rt_cr" {% if state.rt_cr %}checked{% endif %} onchange="sync()"></div>
                             </div>
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
                    <div class="section-header">Alternative Frequencies</div>
                    <div class="section-body">
                         <div class="flex justify-between"><label>Enable AF Method A</label><input type="checkbox" class="toggle-checkbox" id="en_af" {% if state.en_af %}checked{% endif %} onchange="sync()"></div>
                         <textarea id="af_list" rows="2" placeholder="87.5, 98.1, 104.2" onchange="sync()">{{state.af_list}}</textarea>
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

    <script>
        var socket = io();
        var running = {{ 'true' if state.running else 'false' }};
        var pty_list = {{ pty_list|tojson }};
        
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

        function updateRTVisibility() {
            const manual = document.getElementById('rt_manual_buffers').checked;
            document.getElementById('rt_single_mode').style.display = manual ? 'none' : 'block';
            document.getElementById('rt_dual_mode').style.display = manual ? 'block' : 'none';
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
                pi: getVal('pi'), pty: getVal('pty'), tp: getVal('tp'), ta: getVal('ta'), ms: getVal('ms'),
                di_stereo: getVal('di_stereo'), di_head: getVal('di_head'), di_comp: getVal('di_comp'), di_dyn: getVal('di_dyn'), en_af: getVal('en_af'), af_list: getVal('af_list'),
                ps_dynamic: getVal('ps_dynamic'), ps_centered: getVal('ps_centered'),
                rt_text: getVal('rt_text'), rt_manual_buffers: getVal('rt_manual_buffers'), rt_cycle_ab: getVal('rt_cycle_ab'),
                rt_a: getVal('rt_a'), rt_b: getVal('rt_b'), rt_mode: getVal('rt_mode'),
                rt_cycle: getVal('rt_cycle'), rt_centered: getVal('rt_centered'), rt_cr: getVal('rt_cr'),
                rt_cycle_time: getVal('rt_cycle_time'),
                
                // New RT+
                rt_plus_format_a: getVal('rt_plus_format_a'),
                rt_plus_format_b: getVal('rt_plus_format_b'),
                en_rt_plus: getVal('en_rt_plus'),
                
                ptyn: getVal('ptyn'), en_ptyn: getVal('en_ptyn'), ptyn_centered: getVal('ptyn_centered'),
                ecc: getVal('ecc'), lic: getVal('lic'), tz_offset: getVal('tz_offset'), en_ct: getVal('en_ct'), en_id: getVal('en_id'),
                ps_long_32: getVal('ps_long_32'), en_lps: getVal('en_lps'), lps_centered: getVal('lps_centered'),
                en_dab: getVal('en_dab'), dab_channel: getVal('dab_channel'),
                dab_eid: getVal('dab_eid'), dab_mode: getVal('dab_mode'), dab_es_flag: getVal('dab_es_flag'),
                dab_sid: getVal('dab_sid'), dab_variant: getVal('dab_variant'),
                group_sequence: getVal('group_sequence'), scheduler_auto: getVal('scheduler_auto')
            };
            socket.emit('update', data);
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
        }, 1000);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
