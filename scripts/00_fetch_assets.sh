#!/usr/bin/env bash
# Downloads the third-party assets the pipeline needs (fonts, Tesseract, word list).
# Everything goes to data/assets/ (git-ignored); re-running is a no-op.
set -euo pipefail
cd "$(dirname "$0")/.."
A=data/assets
mkdir -p "$A/fonts" "$A/tess/tessdata"

# Cyrillic-capable fonts from google/fonts (OFL / UFL licensed)
G=https://github.com/google/fonts/raw/main
for p in ofl/ptsans/PT_Sans-Web-Regular.ttf ofl/ptsans/PT_Sans-Web-Bold.ttf \
         ofl/ptserif/PT_Serif-Web-Regular.ttf ofl/ptserif/PT_Serif-Web-Bold.ttf \
         ufl/ubuntu/Ubuntu-Regular.ttf ufl/ubuntu/Ubuntu-Bold.ttf \
         ofl/firasans/FiraSans-Regular.ttf ofl/firasans/FiraSans-Bold.ttf \
         "ofl/oswald/Oswald%5Bwght%5D.ttf" "ofl/comfortaa/Comfortaa%5Bwght%5D.ttf" \
         "ofl/caveat/Caveat%5Bwght%5D.ttf" ofl/marckscript/MarckScript-Regular.ttf \
         "ofl/robotocondensed/RobotoCondensed%5Bwght%5D.ttf" "ofl/rubik/Rubik%5Bwght%5D.ttf" \
         "ofl/montserrat/Montserrat%5Bwght%5D.ttf" ofl/lobster/Lobster-Regular.ttf \
         ofl/russoone/RussoOne-Regular.ttf "ofl/notoserif/NotoSerif%5Bwdth,wght%5D.ttf" \
         "ofl/notosans/NotoSans%5Bwdth,wght%5D.ttf" ofl/ptmono/PTM55FT.ttf; do
  n=$(basename "$p" | sed 's/%5B.*//; s/\.ttf//')
  [ -s "$A/fonts/$n.ttf" ] || curl -sSL -o "$A/fonts/$n.ttf" "$G/$p"
done

# Tesseract 5.5.3 (AppImage, extracted) + tessdata_best models
if [ ! -x "$A/tess/squashfs-root/AppRun" ]; then
  curl -sSL -o "$A/tess/t.AppImage" \
    https://github.com/AlexanderP/tesseract-appimage/releases/download/v5.5.3/tesseract-5.5.3_lept-1.87-x86_64.AppImage
  chmod +x "$A/tess/t.AppImage"
  (cd "$A/tess" && ./t.AppImage --appimage-extract >/dev/null)
fi
for l in ukr; do
  [ -s "$A/tess/tessdata/$l.traineddata" ] || curl -sSL -o "$A/tess/tessdata/$l.traineddata" \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/$l.traineddata
done

# Ukrainian word-frequency list (OpenSubtitles 2018, hermitdave/FrequencyWords, CC-BY-SA 4.0)
[ -s "$A/uk_full.txt" ] || curl -sSL -o "$A/uk_full.txt" \
  https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/uk/uk_full.txt
echo "assets ready in $A"
