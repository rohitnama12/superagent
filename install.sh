#!/bin/bash
set -e

# --- Configuration ---
REPO="rohitnama12/superagent"
BINARY_NAME="superagent"
INSTALL_DIR="$HOME/.local/bin"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CYAN}=================================================${NC}"
echo -e "${CYAN}    SuperAgent AI - 1-Click Installer            ${NC}"
echo -e "${CYAN}=================================================${NC}"

# --- 1. Detect OS & Architecture ---
OS="$(uname -s)"
ARCH="$(uname -m)"

echo -e "${BLUE}[*] Detecting system architecture...${NC}"

if [ "$OS" = "Linux" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        ASSET_NAME="superagent-linux-x64"
    else
        echo -e "${RED}[!] Unsupported Linux architecture: $ARCH${NC}"
        exit 1
    fi
elif [ "$OS" = "Darwin" ]; then
    if [ "$ARCH" = "x86_64" ]; then
        ASSET_NAME="superagent-macos-x64"
    elif [ "$ARCH" = "arm64" ]; then
        ASSET_NAME="superagent-macos-arm64"
    else
        echo -e "${RED}[!] Unsupported macOS architecture: $ARCH${NC}"
        exit 1
    fi
# --- ADDED WINDOWS / MINGW / MSYS SUPPORT ---
elif [[ "$OS" =~ MINGW* ]] || [[ "$OS" =~ MSYS* ]] || [[ "$OS" =~ CYGWIN* ]]; then
    ASSET_NAME="superagent-windows-x64.exe"
    BINARY_NAME="superagent.exe"
else
    echo -e "${RED}[!] Unsupported Operating System: $OS${NC}"
    exit 1
fi

echo -e "${GREEN}[+] Detected: $OS $ARCH -> $ASSET_NAME${NC}"

# --- 2. Construct Download URL ---
DOWNLOAD_URL="https://github.com/$REPO/releases/latest/download/$ASSET_NAME"

# --- 3. Download Binary ---
echo -e "${BLUE}[*] Downloading SuperAgent from GitHub...${NC}"
mkdir -p /tmp/superagent_install
curl -sSL -f "$DOWNLOAD_URL" -o "/tmp/superagent_install/$BINARY_NAME" || {
    echo -e "${RED}[!] Failed to download binary. Please ensure $REPO is correct and a release exists.${NC}"
    exit 1
}

# --- 4. Install Binary ---
echo -e "${BLUE}[*] Installing to $INSTALL_DIR...${NC}"
mkdir -p "$INSTALL_DIR"
mv "/tmp/superagent_install/$BINARY_NAME" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/$BINARY_NAME"
rm -rf /tmp/superagent_install

# --- 5. Configure PATH ---
echo -e "${BLUE}[*] Checking PATH configuration...${NC}"

case $SHELL in
*/zsh)
    PROFILE="$HOME/.zshrc"
    ;;
*/bash)
    PROFILE="$HOME/.bashrc"
    ;;
*)
    PROFILE="$HOME/.profile"
    ;;
esac

if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo -e "${YELLOW}[!] $INSTALL_DIR is not in your PATH.${NC}"
    echo -e "${BLUE}[*] Adding $INSTALL_DIR to $PROFILE...${NC}"
    echo -e "\n# Added by SuperAgent Installer" >> "$PROFILE"
    echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$PROFILE"
    echo -e "${GREEN}[+] PATH updated! You will need to restart your terminal or run: source $PROFILE${NC}"
else
    echo -e "${GREEN}[+] $INSTALL_DIR is already in your PATH.${NC}"
fi

# --- 6. Success Message ---
echo -e "${CYAN}=================================================${NC}"
echo -e "${GREEN}   Installation Complete! 🚀                     ${NC}"
echo -e "${CYAN}=================================================${NC}"
echo -e "You can now run SuperAgent from anywhere in your terminal."
echo -e "To trigger the onboarding wizard, simply type:\n"
echo -e "  ${YELLOW}superagent${NC}\n"