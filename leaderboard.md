# Leaderboard

Spearman rank correlation with the human mean score on held-out photos; higher is better. † = the model was
trained on that dataset (or on its photos), so the cell is not a held-out result and is left out of the mean.

## Photo aesthetics

| Model | Mean, unseen sets | para (3,000) | eva (4,070) | tad66k (14,079) | aadb (1,000) | flickr_aes (4,031) | ava (19,929) |
|---|---|---|---|---|---|---|---|
| ours-siglip2-base-ridge | **0.779** (1) | 0.779 | 0.615† | 0.405† | 0.716† | 0.687† | 0.639† |
| ours-siglip2-base-ridge-noava | **0.651** (3) | 0.784 | 0.610 | 0.378† | 0.725† | 0.694† | 0.559 |
| ours-siglip2-base-ridge-lodo | **0.601** (5) | 0.779 | 0.615† | 0.374 | 0.667 | 0.624 | 0.559 |
| topiq_iaa | **0.591** (2) | — | 0.512† | — | 0.571 | 0.611 | 0.791† |
| nima | **0.512** (1) | — | — | — | 0.512 | — | — |
| topiq_iaa_res50 | ? | — | — | — | — | — | — |

## Side papers: technical quality (koniq10k, spaq, paq2piq), paintings (baid, lapis), animal slices

| Model | koniq10k | spaq | paq2piq | baid | lapis | ava:animal | tad66k:bird | para:animal |
|---|---|---|---|---|---|---|---|---|
| ours-siglip2-base-ridge | 0.578 | 0.746 | 0.264 | 0.165 | 0.439 | 0.635† | 0.388† | 0.743 |
| ours-siglip2-base-ridge-noava | 0.566 | 0.725 | 0.280 | 0.159 | 0.387 | 0.566 | 0.377† | 0.734 |
| ours-siglip2-base-ridge-lodo | 0.578 | 0.746 | 0.264 | 0.165 | 0.439 | 0.566 | 0.311 | 0.743 |
| topiq_iaa | — | — | — | 0.050 | — | 0.781† | — | — |
| nima | — | — | — | — | — | — | — | — |
| topiq_iaa_res50 | — | — | — | — | — | — | — | — |

## 95% intervals (bootstrap over photos)

| Model | para | eva | tad66k | aadb | flickr_aes | ava |
|---|---|---|---|---|---|---|
| ours-siglip2-base-ridge | 0.761–0.795 | 0.594–0.636 | 0.391–0.419 | 0.680–0.748 | 0.670–0.706 | 0.630–0.648 |
| ours-siglip2-base-ridge-noava | 0.767–0.799 | 0.589–0.631 | 0.364–0.393 | 0.692–0.758 | 0.676–0.711 | 0.549–0.570 |
| ours-siglip2-base-ridge-lodo | 0.761–0.795 | 0.594–0.636 | 0.360–0.389 | 0.627–0.705 | 0.604–0.646 | 0.549–0.570 |
| topiq_iaa | — | 0.488–0.538 | — | 0.524–0.615 | 0.591–0.633 | 0.785–0.796 |
| nima | — | — | — | 0.458–0.560 | — | — |
| topiq_iaa_res50 | — | — | — | — | — | — |
