#!/bin/bash
# Push to GitHub with retry logic
cd /home/codex/projects/codex_test/fibernet

echo "Pushing to GitHub..."
for i in 1 2 3; do
    echo "Attempt $i..."
    if git push origin main; then
        echo "✓ Push successful!"
        exit 0
    fi
    echo "Attempt $i failed, waiting 10s..."
    sleep 10
done

echo "✗ All push attempts failed. Network issue persists."
echo "To push later: cd /home/codex/projects/codex_test/fibernet && git push origin main"
exit 1
