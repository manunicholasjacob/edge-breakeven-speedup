#!/bin/bash
# Revision campaign runner: one heavy stage at a time, resumable.
cd "$HOME/p15v2" || exit 1
PY="$HOME/multi_constraint_edge_inference/venv/bin/python"
mkdir -p data logs

for stage in idle threads perf ops gemm llm cotenant sustained governor; do
  if [ -f "data/${stage}.DONE" ]; then
    echo "== skip $stage (done)"
    continue
  fi
  echo "== $stage starting $(date -Is)"
  "$PY" campaign.py --stage "$stage" --out "data/E_${stage}.jsonl" \
      >> "logs/${stage}.log" 2>&1
  rc=$?
  echo "== $stage exit=$rc $(date -Is)"
  if [ $rc -eq 0 ]; then
    touch "data/${stage}.DONE"
  fi
done
echo "== campaign complete $(date -Is)"
