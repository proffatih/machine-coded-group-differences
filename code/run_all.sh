#!/bin/bash
# Tam anotasyon kosusu: 6 model x 3 kosul x 7 korpus = 126 shard.
# Yeniden baslatilabilir -- var olan .npz shardlari atlanir, kesinti guvenli.
# Modeller kucukten buyuge sirali: kart dolarsa once buyugunde dolar, kucukler bitmis olur.
set -u
export HF_HOME="$HOME/.cache/huggingface" TMPDIR="$HOME/.cache/pip-tmp" PYTHONUNBUFFERED=1
# Xet arka ucu bu makinede 7 KB/s'ye dusuyor ve kesintiden sonra %91'lik dosyayi
# birakip sifirdan basliyor; duz HTTP 1.8 MB/s ve resume calisiyor.
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="$HOME/Papers/Home_Victus_0010_LLM_Annotation_Calibration/Social_Science_Computer_Review"
cd "$P" || exit 1
LOG="$HOME/.cache/run_all.log"

for M in olmo2-1b qwen2.5-1.5b smollm2-1.7b granite3.1-2b qwen2.5-3b phi3.5-mini; do
  for C in zs cb fs; do
    echo "### $(date +%H:%M:%S) baslıyor: $M / $C" >> "$LOG"
    ./.venv/bin/python code/02_annotate.py --models "$M" --conditions "$C" --batch 4 >> "$LOG" 2>&1
    rc=$?
    n=$(ls results/annotations/*.npz 2>/dev/null | wc -l)
    if [ $rc -ne 0 ]; then
      echo "### $(date +%H:%M:%S) HATA rc=$rc: $M / $C -- atlanip devam ediliyor (shard: $n/126)" >> "$LOG"
    else
      echo "### $(date +%H:%M:%S) tamam: $M / $C (shard: $n/126)" >> "$LOG"
    fi
  done
done
echo "=== TUM ANOTASYON BITTI: $(ls results/annotations/*.npz 2>/dev/null | wc -l)/126 shard ===" >> "$LOG"
./.venv/bin/python code/03_invariance.py >> "$LOG" 2>&1 && echo "=== ANALIZ TAMAM ===" >> "$LOG" || echo "=== ANALIZ HATA ===" >> "$LOG"
