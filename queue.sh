#!/bin/zsh
# Sit every listed contestant, one after another; resumable (run.py appends). Log: ROOT/bench/queue.log
cd /Volumes/Media/aesbench && source env.sh
for spec in "topiq_iaa" "topiq_iaa_res50" "nima" "nima-vgg16-ava" "laion_aes" "musiq-ava" \
            "clipiqa+" "topiq_nr" "musiq" "qrealign-mini --task aesthetic" "qalign --task aesthetic"; do
  for try in 1 2 3; do   # run.py resumes; a clean exit prints "scored"
    echo "== $spec try $try $(date)"
    uv run python -m aesbench.run ${=spec} 2>&1 | tr "\r" "\n" | grep --line-buffered -v "%|\|Warning" | tee /tmp/aesbench-last.log
    grep -q " scored, " /tmp/aesbench-last.log && break
  done
done
echo "QUEUE DONE $(date)"
