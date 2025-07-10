#!/bin/bash

# Feste Werte
TARGET_FILE="/home/tkapler/miniconda3/envs/onthefly_nvs/lib/python3.12/site-packages/graphdecoviewer/widgets/image.py"
SEARCH="imgui.image(self.texture.id, (res_x, res_y))"
REPLACE="imgui.image(imgui.ImTextureRef(self.texture.id), imgui.ImVec2(res_x, res_y))"

# Prüfen ob Datei existiert
if [ ! -f "$TARGET_FILE" ]; then
  echo "❌ Fehler: Datei '$TARGET_FILE' nicht gefunden."
  exit 1
fi

# Prüfen ob der Suchstring existiert
if grep -q "$SEARCH" "$TARGET_FILE"; then
  echo "🔧 Ersetze '$SEARCH' durch '$REPLACE' in '$TARGET_FILE'..."
  sed -i "s/$SEARCH/$REPLACE/g" "$TARGET_FILE"
  echo "✅ Erledigt."
else
  echo "ℹ️ Kein '$SEARCH' in '$TARGET_FILE' gefunden — kein Patch nötig."
fi

