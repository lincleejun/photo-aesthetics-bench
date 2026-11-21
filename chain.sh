#!/bin/zsh
# After the pixel cache is complete: features for every backbone, ridge heads, then mark. Log: ROOT/bench/chain.log
cd /Volumes/Media/aesbench && source env.sh
until grep -q "PIXELS DONE" $AESBENCH_ROOT/bench/pixels.log; do sleep 60; done
for b in siglip2-base dinov2-base clip-l14 bioclip25-b15; do
  echo "== features $b $(date)"
  uv run python -m aesbench.features $b 2>&1 | tr '\r' '\n' | grep --line-buffered -v "%|\|Warning\|warn" | tail -3
  echo "== train $b $(date)"
  uv run python -m aesbench.train $b 2>&1 | tail -4
done
uv run python -m aesbench.score > /dev/null && echo "CHAIN DONE $(date)"
