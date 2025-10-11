#!/bin/bash
# Download sample SGF files for training
# This script helps you get started with supervised learning from pro games

set -e

echo "================================================"
echo "  SGF Game Downloader for Go Training"
echo "================================================"
echo ""

# Create directory for SGF files
SGF_DIR="${1:-./go_games_9x9}"
mkdir -p "$SGF_DIR"

echo "Downloading sample SGF files to: $SGF_DIR"
echo ""

# Note: This is a template script. You'll need to add actual sources.
# Here are some options to get SGF files:

echo "Option 1: Download from GoGoD (paid, high quality)"
echo "  Visit: https://gogodonline.co.uk/"
echo "  Download and extract to: $SGF_DIR"
echo ""

echo "Option 2: Download from KGS Archives (free)"
echo "  Visit: https://www.gokgs.com/gameArchives.jsp"
echo "  1. Select a date range"
echo "  2. Filter by rank (5d+ recommended)"
echo "  3. Download SGF files"
echo "  4. Extract to: $SGF_DIR"
echo ""

echo "Option 3: Use a GitHub dataset"
echo "  Example repositories with Go game collections:"
echo "  - https://github.com/yenw/computer-go-dataset"
echo "  - https://github.com/featurecat/go-dataset"
echo ""

# Example: Download a small sample if available
# This is a placeholder - replace with actual source
echo "Checking for publicly available sample datasets..."

# Try to download from a hypothetical public source
# (Replace this with actual working URLs)
if command -v wget &> /dev/null; then
    echo "wget is available for downloading"
    # wget -P "$SGF_DIR" "http://example.com/sample_9x9_games.tar.gz"
    # tar -xzf "$SGF_DIR/sample_9x9_games.tar.gz" -C "$SGF_DIR"
elif command -v curl &> /dev/null; then
    echo "curl is available for downloading"
    # curl -o "$SGF_DIR/sample_9x9_games.tar.gz" "http://example.com/sample_9x9_games.tar.gz"
    # tar -xzf "$SGF_DIR/sample_9x9_games.tar.gz" -C "$SGF_DIR"
else
    echo "Neither wget nor curl found. Please download SGF files manually."
fi

echo ""
echo "================================================"
echo "Manual Download Instructions:"
echo "================================================"
echo ""
echo "1. Visit one of the sources above"
echo "2. Download SGF game files (preferably 9x9 for your training)"
echo "3. Place them in: $SGF_DIR"
echo "4. Then run:"
echo ""
echo "   python train_from_sgf.py --sgf_dir $SGF_DIR --board_size 9"
echo ""
echo "For best results:"
echo "  - Get 1000+ games"
echo "  - Filter for high-rank players (dan level)"
echo "  - Make sure they're for the correct board size (9x9)"
echo ""

