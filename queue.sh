#!/bin/zsh
# Sit the contestants given as arguments, one after another; resumable (run.py appends). Log: stdout.
#   ./queue.sh "nima" "laion_aes"                       # a lane; run a second lane beside it for other contestants
#   ./queue.sh                                          # everyone
# Small models wait on the disk, the two language models on the GPU: one lane of each runs well side by side.
cd /Volumes/Media/aesbench && source env.sh
(( $# )) || set -- "topiq_iaa" "topiq_iaa_res50" "nima" "nima-vgg16-ava" "laion_aes" "musiq-ava" \
                   "clipiqa+" "topiq_nr" "musiq" "qrealign-mini --task aesthetic" "qalign --task aesthetic"
last=$(mktemp)
for spec in "$@"; do
  for try in 1 2 3; do   # run.py resumes; a clean exit prints "scored"
    echo "== $spec try $try $(date)"
    uv run python -m aesbench.run ${=spec} 2>&1 | tr "\r" "\n" | grep --line-buffered -v "%|\|Warning" | tee $last
    grep -q " scored, " $last && break
  done
done
echo "QUEUE DONE $(date)"
