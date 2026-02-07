

# RDS Master Pro (Python)

  ![Overview of the User Interface](http://uploads.mpbnl.nl/u/LeTzSM.png)

A work-in-progress open-source webUI based RDS encoder. Ships with a lightweight Flask + Socket.IO server, Tailwind-styled UI, and session-gated access for secure usage.

Many thanks to **Bkram, Hans van Eijsden, RZCH, Wötkylä, notluca, Hyper DX and Adam W** (not in order) for Testing, ideas and assistance in the development of this program! ❤️

  

## Current Features

- Web UI with tabs for Dashboard, Basic RDS, Expert, Audio, and Settings.

- Config-driven login (credentials stored in `config.ini` under `[AUTH]`).

- Auto-start option (enabled by default) to bring the encoder on-air when the moment the program launches; toggle in Settings.

- Device selection for input/output (only MME devices on Windows at the moment) plus pass-through and genlock controls (uses 19kHz carrier for RDS carrier phasing).

- Dynamic PS/RT editing with simple timed sequencing and centering options.

- RT+ formatting with visual builder and support for up to 2 tags per RT message.

- Long PS (32 chars), PTYN, CT, AF Method A & B controls; DAB cross-reference (12A) (experimental).

- Enhanced Other Networks (EON) Group 14A support with PS/AF/PTY transmission (experimental, TA/TP not working).
- FM-DAB linking via ODA
- Manual RDS group input from RDS Spy recording / Airomate-standard txt file 

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

- RT+ tagging with visual builder supporting 2 tags per message. ✅

- EON (Enhanced Other Networks) Group 14A implemented but **experimental** - use with caution and verify on-air behavior. ⚠️

- AF Method B support. ✅

- DAB cross-reference (Group 12A) is experimental, verified on-air. ✅

- Extended ASCII character support (ISO-8859-1/Latin-1 encoding). ✅

- Datasets Mode (Group 5A) for transparent data channel. ✅

- No packaged EXE release yet

- Support for RBDS variant. ✅

- Support for UECP Input / output (planned)

- Support for MRDS1322 RDS encoder chip (planned)

## Development Status

Active development; breaking changes are possible. Avoid production deployment until a stable release is published. Feedback, contributions and issue reports are welcome.

Déanta in Éirinn 💖🇮🇪
