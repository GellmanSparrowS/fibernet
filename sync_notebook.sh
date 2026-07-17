#!/bin/bash
# Sync notebook between repo and /media/sf_share
REPO="/home/codex/projects/codex_test/fibernet/tutorials/fibernet_v4_tutorial_updated.ipynb"
SHARE="/media/sf_share/fibernet_v4_tutorial_updated.ipynb"

if [ "$1" == "to_share" ]; then
    cp "$REPO" "$SHARE"
    echo "✓ Copied repo → share"
elif [ "$1" == "to_repo" ]; then
    cp "$SHARE" "$REPO"
    echo "✓ Copied share → repo"
else
    echo "Usage: $0 [to_share|to_repo]"
fi
