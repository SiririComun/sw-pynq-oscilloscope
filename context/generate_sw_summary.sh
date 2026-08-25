#!/usr/bin/env bash
# ==============================================================================
# Script: context/generate_sw_summary.sh
# Target: sw-pynq-oscilloscope
# Purpose: Dump Git history, structure, SW files, and notebook JSON into context/sw_summary.txt
# ==============================================================================

# Dynamically resolve paths
CONTEXT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CONTEXT_DIR}/.." && pwd)"
OUTPUT_FILE="${CONTEXT_DIR}/sw_summary.txt"

echo "Generating concise SW summary into: ${OUTPUT_FILE}..."
> "${OUTPUT_FILE}"

cd "${REPO_ROOT}" || exit 1

# 1. Header, Branch Status & Git History
cat << 'EOF' >> "${OUTPUT_FILE}"
Git Context & Branch Info:

```log
EOF

if command -v git &> /dev/null && git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Current Branch: $(git branch --show-current)" >> "${OUTPUT_FILE}"
    echo "Branch Tracking Status:" >> "${OUTPUT_FILE}"
    git branch -vv >> "${OUTPUT_FILE}"
    echo "" >> "${OUTPUT_FILE}"
    echo "Git Commit Graph & History:" >> "${OUTPUT_FILE}"
    git log --graph --pretty=format:"%h %d - %cd : %s (%an)" --date=short -n 15 >> "${OUTPUT_FILE}"
fi

cat << 'EOF' >> "${OUTPUT_FILE}"
```

Files structure: 

```log
EOF

# 2. Directory Tree
if command -v tree &> /dev/null; then
    tree -a -I '.git|__pycache__|*.egg-info|.pytest_cache|venv|env|dist|build|.ipynb_checkpoints' >> "${OUTPUT_FILE}"
fi

cat << 'EOF' >> "${OUTPUT_FILE}"
```

EOF

# 3. Exact list of targeted files and their markdown syntax
TARGET_FILES=(
    ".github/workflows/pypi.yml:yaml"
    "hardware.json:json"
    "setup.py:python"
    "requirements.txt:text"
    "pynq_oscilloscope/__init__.py:python"
    "pynq_oscilloscope/dashboard.py:python"
    "pynq_oscilloscope/env_checker.py:python"
    "pynq_oscilloscope/loader.py:python"
    "pynq_oscilloscope/notebooks.py:python"
    "pynq_oscilloscope/xadc_dma.py:python"
    "pynq_oscilloscope/fft_dma.py:python"
    "pynq_oscilloscope/ad3_wavegen.py:python"
    "pynq_oscilloscope/hw_trigger.py:python"
    "pynq_oscilloscope/overlay.py:python"
    "pynq_oscilloscope/analytics.py:python"
    "pynq_oscilloscope/audio_utils.py:python"
    "pynq_oscilloscope/analytic_dashboard.py:python"
    "notebooks/01_ad3_getting_started.ipynb:json"
    "notebooks/02_xadc_getting_started.ipynb:json"
    "notebooks/03_oscilloscope_dashboard.ipynb:json"
    "notebooks/04_audio_dashboard.ipynb:json"
    "notebooks/05_acoustic_analytic_curves.ipynb:json"
    "notebooks/06_audio_recording_and_playback.ipynb:json"
    "context/generate_sw_summary.sh:bash"
)

# 4. Append each file
for item in "${TARGET_FILES[@]}"; do
    filepath="${item%%:*}"
    syntax="${item##*:}"
    filename=$(basename "$filepath")

    if [ -f "$filepath" ]; then
        echo "Adding: ${filepath}"
        {
            echo "${filename}:"
            echo ""
            echo "\`\`\`${syntax}"
            cat "${filepath}"
            echo "\`\`\`"
            echo ""
        } >> "${OUTPUT_FILE}"
    else
        echo "Warning: File not found -> ${filepath}"
    fi
done

echo "Done! SW context generated in: ${OUTPUT_FILE}"