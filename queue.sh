#!/bin/zsh
# Sit every listed contestant, one after another; resumable (run.py appends). Log: ROOT/bench/queue.log
cd /Volumes/Media/aesbench && source env.sh
for spec in "topiq_iaa" "topiq_iaa_res50" "nima" "nima-vgg16-ava" "laion_aes" "musiq-ava" \
            "clipiqa+" "topiq_nr" "musiq" "qrealign-mini --task aesthetic" "qalign --task aesthetic"; do
  echo "== $spec $(date)"; uv run python -m aesbench.run ${=spec} 2>&1 | tr '\r' '\n' | grep -v "%|\|Warning"
done
echo "QUEUE DONE $(date)"
