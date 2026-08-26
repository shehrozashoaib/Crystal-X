#!/bin/bash
# watch_job.sh <jobid> <logfile> — poll until the job leaves the queue, then summarize.
JOB="$1"; LOG="$2"
echo "[watch] job=$JOB log=$LOG start=$(date -Iseconds)"
while squeue -h -j "$JOB" -o "%T" 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
  sleep 60
done
echo "[watch] job $JOB left the queue at $(date -Iseconds)"
echo "[watch] sacct:"; sacct -j "$JOB" --format=JobID,JobName%20,State,Elapsed,MaxRSS,ExitCode 2>/dev/null | head -8
echo "[watch] ===== tail of $LOG ====="
tail -25 "$LOG" 2>/dev/null || echo "  (log not found)"
echo "[watch] ===== OK/FAIL markers ====="
grep -aE "smoke OK|smoke FAILED|Test MAE|test MAE|test metric|bestfix|Error|Traceback|CUDA out of memory" "$LOG" 2>/dev/null | tail -25
