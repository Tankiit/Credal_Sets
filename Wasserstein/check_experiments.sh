#!/bin/bash
# Check status of all running experiments

echo "========================================"
echo "EXPERIMENT STATUS CHECK"
echo "========================================"
echo ""

# Check processes
echo "Running processes:"
pgrep -fl "train_ternary_50epochs.py" | wc -l | xargs echo "  Total experiments running:"
echo ""

# Show details
pgrep -fl "train_ternary_50epochs.py" | while read line; do
  echo "  $line"
done
echo ""

# Check logs
echo "========================================"
echo "LOG FILE STATUS"
echo "========================================"
echo ""

for dir in experiments/ternary_dro_comparison/*/; do
  name=$(basename "$dir")
  log="$dir/training.log"

  if [ -f "$log" ]; then
    # Get last line with epoch info
    last_epoch=$(grep -o "Epoch [0-9]*/50" "$log" | tail -1)
    if [ -z "$last_epoch" ]; then
      last_epoch=$(grep -o "Epoch [0-9]*/" "$log" | tail -1)
    fi

    # Check if completed
    if grep -q "TRAINING AND EVALUATION COMPLETE" "$log"; then
      status="✅ COMPLETED"
    elif grep -q "Error\|Traceback\|Exception" "$log"; then
      status="❌ ERROR"
    else
      status="🔄 RUNNING"
    fi

    # File size
    size=$(du -h "$log" | cut -f1)

    echo "[$status] $name ($size) - $last_epoch"
  else
    echo "[⚠️  NO LOG] $name"
  fi
done

echo ""
echo "To follow logs in real time:"
echo "  tail -f experiments/ternary_dro_comparison/*/training.log"
