#!/bin/bash

# Directory for sounds
SOUND_DIR="$(dirname "$0")"
mkdir -p "$SOUND_DIR"

echo "Downloading piano samples..."

BASE_URL="https://github.com/shanealder/Piano/raw/master/Samples"

# Map C through B to the corresponding sample files in the repo
# Mapping to 4th octave samples (C4, D4, etc.)
declare -A notes=( ["C"]="C4.wav" ["D"]="D4.wav" ["E"]="E4.wav" ["F"]="F4.wav" ["G"]="G4.wav" ["A"]="A4.wav" ["B"]="B4.wav" )

for note in "${!notes[@]}"; do
    filename="${notes[$note]}"
    target="$SOUND_DIR/$note.wav"
    
    echo "Downloading $note ($filename) -> $target"
    curl -L -s "$BASE_URL/$filename" -o "$target"
done

echo "Done. All samples downloaded to $SOUND_DIR"
