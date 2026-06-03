#!/bin/bash
#
# OPTIONAL: download real piano samples from an external source.
#
# For a guaranteed-working, offline set of samples use generate_samples.py instead:
#     python3 generate_samples.py
#
# This script is only useful if you have a VALID BASE_URL that serves raw .wav
# files. The previous default URL (shanealder/Piano) is dead and returned HTML
# 404 pages, which is how broken "samples" ended up committed. This version uses
# `curl -f`, checks the HTTP status, and verifies each download is a real RIFF
# WAV before keeping it — so a bad URL fails loudly instead of saving garbage.

set -euo pipefail

SOUND_DIR="$(cd "$(dirname "$0")" && pwd)"

# >>> Set this to a source that serves raw C4.wav, D4.wav, ... files. <<<
BASE_URL="${BASE_URL:-https://github.com/shanealder/Piano/raw/master/Samples}"

echo "Downloading piano samples from: $BASE_URL"

declare -A notes=( ["C"]="C4.wav" ["D"]="D4.wav" ["E"]="E4.wav" ["F"]="F4.wav" ["G"]="G4.wav" ["A"]="A4.wav" ["B"]="B4.wav" )

failed=0
for note in "${!notes[@]}"; do
    filename="${notes[$note]}"
    target="$SOUND_DIR/$note.wav"
    tmp="$(mktemp)"

    echo "  $note ($filename) -> $target"
    if ! curl -fL -s "$BASE_URL/$filename" -o "$tmp"; then
        echo "    ERROR: download failed (HTTP error or unreachable)."
        rm -f "$tmp"; failed=1; continue
    fi

    # Reject anything that is not a real RIFF/WAVE file (e.g. an HTML error page).
    if ! head -c 4 "$tmp" | grep -q "RIFF"; then
        echo "    ERROR: downloaded file is not a WAV (got $(file -b "$tmp")). Skipping."
        rm -f "$tmp"; failed=1; continue
    fi

    mv "$tmp" "$target"
done

if [ "$failed" -ne 0 ]; then
    echo
    echo "Some samples failed. Set a valid BASE_URL, or just run:  python3 generate_samples.py"
    exit 1
fi

echo "Done. All samples saved to $SOUND_DIR"
