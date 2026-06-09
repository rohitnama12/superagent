# SuperAgent AI 🤖

SuperAgent is an autonomous Context Engine & Data Mover CLI application designed to automate intelligence tasks across your operating system and files. It's built in Python and runs seamlessly across Windows, macOS, and Linux.

## 🚀 Quick Start & Installation

You don't need to install Python or set up a virtual environment! SuperAgent is packaged as a single standalone executable.

### Mac & Linux (1-Click Install)

Run the following command in your terminal to download and install the latest SuperAgent release:

```bash
curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/install.sh | bash
```

Once installed, simply run `superagent` from anywhere in your terminal to trigger the onboarding wizard!

### Windows Installation

1. Go to the [Releases Tab](https://github.com/YOUR_USERNAME/YOUR_REPO/releases/latest) on GitHub.
2. Download `superagent-windows-x64.exe`.
3. Move the downloaded `.exe` file to a permanent folder (e.g., `C:\Program Files\SuperAgent\`).
4. **Add to PATH**:
   - Open your Start Menu and search for "Environment Variables".
   - Click "Edit the system environment variables".
   - Under the "Advanced" tab, click "Environment Variables...".
   - Under "System variables" (or "User variables"), find the `Path` variable, select it, and click "Edit".
   - Click "New" and add the path to the folder where you saved the `.exe` (e.g., `C:\Program Files\SuperAgent\`).
   - Click OK to save.
5. Open a new Command Prompt or PowerShell and type `superagent-windows-x64.exe` to start the agent!

## Features
- **Local Fallback Mode**: Works with Local Ollama models.
- **Dynamic Semantic Search (RAG)**: Uses ChromaDB to index your local repository for semantic understanding.
- **Document Parsing**: Native support for PDF, DOCX, and OCR scanning.
- **Git Checkpoints**: Automatically creates revertible checkpoints before modifying files.

---

*Note: SuperAgent creates local files such as `.env` and `.chroma_db` locally inside the working directory when executed to store API keys and Vector DB indexes respectively.*
