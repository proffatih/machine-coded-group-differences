#!/usr/bin/env bash
# 7-8B modelleri 8 GB kartta (RTX 3070) kodlar, shard'lari bu repoya push eder.
# Kullanim:  bash run_large_models.sh                      # uc modelin hepsi (tek kart)
#            bash run_large_models.sh qwen2.5-7b            # tek model (iki kartta bolmek icin)
# Yarida kalirsa ayni komutu tekrar calistir: hazir shard'lar atlanir.
set -euo pipefail
cd "$(dirname "$0")"
MODELS=("${@:-qwen2.5-7b olmo2-7b granite3.1-8b}")
git pull --no-rebase --no-edit
python3 -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))"
# once kisa deneme: 8 ogeyle bellek/akis kontrolu (shard yazmaz)
for m in ${MODELS[@]}; do
  python3 code/02_annotate.py --models "$m" --conditions fs --corpora us_nyt --limit 8 --batch 4 \
    >/dev/null 2>&1 && echo "[$m] deneme OK" || { echo "[$m] DENEME BASARISIZ"; exit 1; }
  rm -f results/annotations/"$m"__fs__us_nyt.npz
done
mkdir -p logs
for m in ${MODELS[@]}; do
  python3 code/02_annotate.py --models "$m" 2>&1 | tee -a "logs/annotate_$m.log"
  git add results/annotations/"$m"__*.npz
  git commit -m "Annotate with $m (7-8B, NF4)" || true
  git pull --no-rebase --no-edit && git push
done
echo "BITTI: ${MODELS[*]}"
