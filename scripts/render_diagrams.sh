#!/usr/bin/env bash
# Renders all Mermaid diagrams from docs/architecture.md into docs/diagrams/
set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR/.."

PUPPETEER_CONFIG="$SCRIPT_DIR/puppeteer-config.json"

# Install mermaid-cli if not available
if ! command -v mmdc &>/dev/null; then
    echo "Installing @mermaid-js/mermaid-cli..."
    npm install -g @mermaid-js/mermaid-cli
fi

mkdir -p docs/diagrams
rm -f docs/diagrams/*.mmd

# Extract mermaid code blocks from architecture.md and render each one
awk '
/^## [0-9]+\. /{
    # Extract diagram name from heading: "## 1. High-Level System Overview" -> "01_high_level_system_overview"
    title = $0
    sub(/^## [0-9]+\. /, "", title)
    gsub(/[^a-zA-Z0-9 ]/, "", title)
    gsub(/ +/, "_", title)
    tolower(title)
    num = $2
    sub(/\./, "", num)
    name = sprintf("%02d_%s", num, tolower(title))
}
/^```mermaid$/{
    capturing = 1
    file = "docs/diagrams/" name ".mmd"
    next
}
/^```$/ && capturing {
    capturing = 0
    next
}
capturing {
    print >> file
}
' docs/architecture.md

echo "Extracted diagrams:"
ls docs/diagrams/*.mmd

echo ""
echo "Rendering to PNG..."
for mmd in docs/diagrams/*.mmd; do
    base=$(basename "$mmd" .mmd)
    out="docs/diagrams/${base}.png"
    echo "  $mmd -> $out"
    mmdc -i "$mmd" -o "$out" -t default -b white -w 1600 -p "$PUPPETEER_CONFIG"
done

echo ""
echo "Rendering to SVG..."
for mmd in docs/diagrams/*.mmd; do
    base=$(basename "$mmd" .mmd)
    out="docs/diagrams/${base}.svg"
    echo "  $mmd -> $out"
    mmdc -i "$mmd" -o "$out" -t default -b white -w 1600 -p "$PUPPETEER_CONFIG"
done

echo ""
echo "Done! Files in docs/diagrams/"
ls -la docs/diagrams/
