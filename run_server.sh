#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${DIR}"

echo "========================================================"
echo "⚡ KHỞI ĐỘNG AI HOUSE PRICE PREDICTION SERVER TRÊN COLAB"
echo "========================================================"

# 1. Cài đặt các thư viện cần thiết
echo ">> [1/5] Kiểm tra và cài đặt dependencies..."
python3 -m pip install -q --break-system-packages -r requirements.txt > /dev/null 2>&1 || true

# 2. Huấn luyện hoặc cập nhật mô hình nếu cần
echo ">> [2/5] Huấn luyện / kiểm tra mô hình AI Linear Regression..."
python3 -m app.ml.train

# 3. Dọn dẹp tiến trình cũ trên port 8000
echo ">> [3/5] Dọn dẹp tiến trình cũ trên cổng 8000..."
fuser -k 8000/tcp > /dev/null 2>&1 || true
pkill -f 'uvicorn.*8000' > /dev/null 2>&1 || true
tmux kill-session -t house_price_app 2>/dev/null || true
sleep 1

# 4. Khởi chạy Uvicorn trong background (tmux)
echo ">> [4/5] Khởi chạy FastAPI Uvicorn trên 0.0.0.0:8000..."
tmux new-session -d -s house_price_app "cd ${DIR} && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> ${DIR}/server.log 2>&1"
sleep 2

# 5. Khởi chạy Cloudflare Tunnel công khai
echo ">> [5/5] Cấu hình Cloudflare Public Tunnel..."
if ! command -v cloudflared &> /dev/null; then
    echo ">> Đang tải cloudflared..."
    curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

if ! pgrep -x cloudflared > /dev/null; then
    tmux kill-session -t cf_tunnel 2>/dev/null || true
    tmux new-session -d -s cf_tunnel "cloudflared tunnel --url http://127.0.0.1:8000 > /tmp/cf_tunnel.log 2>&1"
    sleep 4
fi

echo ">> Hoàn tất khởi động! Kiểm tra log tại ${DIR}/server.log"
