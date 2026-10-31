#!/bin/bash
# 01 - NVIDIA driver for Tesla P100 on Debian 13.
# 580 is the last branch that still runs Pascal; CUDA 12.9 is the last toolkit that
# compiles for sm_60. Do not go above either.
set -euo pipefail

DRIVER=${DRIVER:-580.196.07}
URL="https://download.nvidia.com/XFree86/Linux-x86_64/${DRIVER}/NVIDIA-Linux-x86_64-${DRIVER}.run"

echo "== prerequisites =="
apt-get update
apt-get install -y linux-headers-amd64 build-essential dkms curl

echo "== download ${DRIVER} =="
cd /root
[ -f "NVIDIA-Linux-x86_64-${DRIVER}.run" ] || curl -fsSL -o "NVIDIA-Linux-x86_64-${DRIVER}.run" "$URL"
sh "NVIDIA-Linux-x86_64-${DRIVER}.run" --check

echo "== blacklist nouveau and reboot =="
cat > /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF
update-initramfs -u
echo "reboot now, then run this script again with --install"
if [ "${1:-}" != "--install" ]; then
    echo "  (skipping the install; the module cannot bind while nouveau holds the cards)"
    exit 0
fi

cd /root
sh "NVIDIA-Linux-x86_64-${DRIVER}.run" --silent --dkms --no-x-check --no-nouveau-check --disable-nouveau
nvidia-smi
echo "== keep the GPUs initialized, and let nvidia-drm work later =="
printf 'options nvidia-drm modeset=1\noptions nvidia NVreg_PreserveVideoMemoryAllocations=1\n' \
    > /etc/modprobe.d/nvidia-drm.conf
update-initramfs -u
