#!/bin/zsh
# All backbones at once, each following the pixel cache as it grows; a head is trained as soon as its features are complete.
# Log: ROOT/bench/chain.log (+ chain-<backbone>.log)
cd /Volumes/Media/aesbench && source env.sh
B=$AESBENCH_ROOT/bench
N=$(($(wc -l < $B/manifest.csv) - 1))
for b in siglip2-base dinov2-base clip-l14 bioclip25-b15; do
  ( while [[ $(cat $B/features/$b.done 2>/dev/null || echo 0) -lt $N ]]; do
      uv run python -m aesbench.features $b 2>&1 | tr '\r' '\n' | grep --line-buffered -v "%|\|Warning\|warn" | tail -1
      [[ $(cat $B/features/$b.done) -lt $N ]] && sleep 120
    done
    echo "== train $b $(date)"; uv run python -m aesbench.train $b 2>&1 | tail -4
    uv run python -m aesbench.score > /dev/null; echo "== scored after $b $(date)"
  ) > $B/chain-$b.log 2>&1 &
done
wait
uv run python -m aesbench.score > /dev/null && echo "CHAIN DONE $(date)"
