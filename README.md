
# SuperAgent 🤖⚡

> **An Enterprise-Grade, Cross-Platform AI Agent with Hybrid Document Parsing & Surgical OCR.**

SuperAgent is an autonomous CLI developer assistant engineered for complex technical workflows, precise document parsing, and local execution. Built with an adaptive hybrid parsing engine, local vector storage, and multi-device hardware acceleration (Apple Silicon MPS / NVIDIA CUDA / CPU), SuperAgent runs natively across macOS and Windows with minimal setup friction.

---

## 🌟 Key Features

* **Adaptive Hybrid Parsing:** Intelligently combines fast structural text extraction (`pdfplumber` / `PyMuPDF`) with selective, high-precision OCR (`EasyOCR`) for scanned documents and image regions.
* **Hardware Accelerated Inference:** Automatically detects host architecture to leverage Apple Silicon MPS (Metal) on Mac or CUDA on Windows/Linux.
* **Granular Observability:** Real-time terminal UI powered by `rich` featuring concurrent tool execution spinners, deterministic event timelines, and session workspace metrics.
* **Zero-Friction First Run:** Automated local model downloading (`SentenceTransformers` & `EasyOCR`) and interactive API Key validation for OpenRouter & Wavily.
* **Cross-Platform Binary Distribution:** Runs as a standalone, zero-dependency executable on both macOS and Windows.

---

## 🚀 Quick Start (Installation)

Installing SuperAgent takes less than 30 seconds. Choose the installation method for your platform:

### 1. Universal One-Line Installer (Recommended)

Run the universal installer script in your terminal. It automatically detects your operating system and architecture, downloads the latest compiled binary, and links it to your system path:

```bash
curl -sSL https://raw.githubusercontent.com/rohitnama12/superagent/main/install.sh | bash

```

### 2. Manual Binary Download

If you prefer downloading directly:

1. Navigate to the **[Releases Tab](https://www.google.com/search?q=https://github.com/rohitnama/superagent/releases)**.
2. Download the appropriate executable for your OS:
* **macOS (Apple Silicon / Intel):** `superagent-macos`
* **Windows (x64):** `superagent-windows-x64.exe`


3. Add the binary to your system PATH or execute it directly from your terminal.

---

## ⚙️ Initial Configuration

When you run `superagent` for the first time, an interactive setup wizard will guide you through adding your API credentials and setting up models:

```text
🚀 Welcome to SuperAgent! Initializing first-time setup...

Enter your OpenRouter API Key : ********************
Enter your Wavily API Key     : ********************
Selecting default model       : openrouter/owl-alpha (Free Tier Verified)

✅ Credentials validated successfully!
⏳ Downloading local embedding & OCR models (First run only)...
✅ All systems initialized.

```

### Managing Credentials & Models

To update your API keys or change your default model at any time, run:

```bash
superagent change_keys

```

---

## 💻 Developer Setup & Local Building

If you are a developer looking to contribute, inspect the codebase, or build from source, follow these steps:

### Prerequisites

* Python 3.10+
* [Poetry](https://python-poetry.org/) (Dependency & Virtual Environment Manager)

### Local Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/rohitnama/superagent.git
cd superagent

```


2. **Install Dependencies:**
```bash
poetry install

```


3. **Run Locally:**
```bash
poetry run superagent

```


4. **Build Executable (PyInstaller):**
```bash
poetry run pyinstaller --onefile superagent/cli.py --name superagent

```



---

## 🛠️ System Architecture

```text
                     +----------------------------------+
                     |           SuperAgent             |
                     |         CLI Interface            |
                     +----------------------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
+------------------+                                     +--------------------+
| Hybrid Parsing   |                                     | Execution Engine   |
| Engine           |                                     | & Observability    |
+------------------+                                     +--------------------+
| - Structural PDF |                                     | - Async Tool Gather|
| - Surgical OCR   |                                     | - Latency Tracker  |
| - Device Router  |                                     | - Rich UI Status   |
+------------------+                                     +--------------------+
         |                                                         |
         +----------------------------+----------------------------+
                                      |
                     +----------------------------------+
                     |      Local Storage & Models      |
                     |      (~/.superagent/models/)     |
                     +----------------------------------+

```

---

## 📜 License & Usage Policy

Copyright (c) 2026 Rohit Nama. All Rights Reserved.

This project is made publicly available for educational, auditing, and portfolio showcase purposes. Binaries are compiled and distributed for personal developer use. Commercial redistribution or re-licensing without permission is prohibited.

---

### Questions or Feedback?

Feel free to open an issue or start a discussion in the **Issues** tab!oma_db` locally inside the working directory when executed to store API keys and Vector DB indexes respectively.*
