
# RDS Master Pro (Python)

  ![Overview of the User Interface](http://uploads.mpbnl.nl/u/LeTzSM.png)

A work-in-progress open-source webUI based RDS encoder. Ships with a lightweight Flask + Socket.IO server, Tailwind-styled UI, and session-gated access for secure usage.

  

## Current Features

- Web UI with tabs for Dashboard, Basic RDS, Expert, Audio, and Settings.

- Config-driven login (credentials stored in `config.ini` under `[AUTH]`).

- Auto-start option (enabled by default) to bring the encoder on-air when the moment the program launches; toggle in Settings.

- Device selection for input/output (only MME devices on Windows at the moment) plus pass-through and genlock controls (uses 19kHz carrier for RDS carrier phasing).

- Dynamic PS/RT editing with simple timed sequencing and centering options.

- RT+ formatting for basic `{artist} - {title}` style extraction.

- Long PS, PTYN, CT, and AF Method A controls; DAB cross-reference (12A) (experimental!).

- Live monitor panel that reflects PS/RT/PI/PTY and pilot status via WebSocket.

  

## Installation

1. Install Python 3.10+ and the PortAudio-compatible drivers for your audio hardware.

2. Install dependencies:

```bash

pip install flask flask-socketio sounddevice numpy scipy

```

  

## Usage

1. Start the app:

```bash

python app.py

```

2. Browse to `http://localhost:5000` and log in with the credentials from `config.ini` ( defaults are`user` / `pass`).

3. In **Settings**, adjust:

- Auto-start on launch (default on).

- Username/password (leave password blank to keep the current one).

4. Select audio devices in **Audio & MPX**, set RDS fields in **Basic/Expert**, then toggle **ON AIR**.

5. Config is persisted back to `config.ini` on changes or when stopping the encoder.

  

## Limits and Roadmap

- Not production-ready; use only for lab/bench testing until a formal release.

- RT+ tagging is basic (simple artist/title extraction only).

- EON (Enhanced Other Networks) not implemented (planned).

- AF Method B not implemented (planned; only AF Method A available).

- DAB cross-reference is experimental and unverified on-air.

- No packaged EXE release yet; restart-on-crash not managed.

- Support for RBDS

  

## Development Status

Active development; breaking changes are possible. Avoid production deployment until a stable release is published. Feedback, contributions and issue reports are welcome.

Déanta in Éirinn 💖🇮🇪
