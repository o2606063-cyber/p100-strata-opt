#!/bin/bash
# 10-minute whole-machine burn-in: CPU + RAM + IO + all 3 GPUs.
export LC_ALL=C LANG=C
DUR=${1:-600}
LOG=/home/yu/burnin.log
: > "$LOG"

echo "=== burn-in start $(date +%H:%M:%S) for ${DUR}s ===" >> "$LOG"

# GPU burn: all devices, 2 GB each
/home/yu/burn "$DUR" 2048 > /home/yu/gpu-burn.log 2>&1 &
GPU_PID=$!

# CPU + RAM + IO + disk
stress-ng --cpu 16 --cpu-method all \
          --vm 6 --vm-bytes 2G --vm-method all \
          --io 4 --hdd 2 --hdd-bytes 4G \
          --timeout "${DUR}s" --metrics-brief > /home/yu/stress-ng.log 2>&1 &
STRESS_PID=$!

# sampler
i=0
while [ $i -lt $((DUR / 10)) ]; do
  {
    echo "--- $(date +%H:%M:%S)"
    nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,power.draw,clocks.sm,memory.used \
      --format=csv,noheader | sed 's/^/GPU /'
    awk '/MemTotal|MemAvailable|SwapFree/ {printf "MEM %s %.1f GiB\n", $1, $2/1048576}' /proc/meminfo
    awk '{printf "LOAD %.2f\n", $1}' /proc/loadavg
    /usr/local/bin/fans 2>/dev/null | sed -n '2,5p' | sed 's/^/FAN /'
    awk '{print "CPU_TEMP "$1}' /sys/class/hwmon/hwmon5/temp1_input 2>/dev/null
  } >> "$LOG"
  i=$((i + 1))
  sleep 10
done

wait $GPU_PID $STRESS_PID 2>/dev/null
echo "=== burn-in end $(date +%H:%M:%S) ===" >> "$LOG"
cat /home/yu/gpu-burn.log >> "$LOG"
grep -E 'metrc|stress-ng: info' /home/yu/stress-ng.log | tail -5 >> "$LOG"
