#!/usr/bin/env bash
set -e

# ==============================================================================
# Script CI/CD Tự Động Triển Khai Dự Án AI Dự Đoán Giá Nhà Lên Colab Server
# ==============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_IP="${1:-100.91.70.122}"
REMOTE_USER="root"
SSH_KEY="${HOME}/.ssh/colab_ci_cd"
REMOTE_DIR="/content/house_price_prediction_ai"

echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}   🚀 CI/CD DEPLOYMENT: AI HOUSE PRICE PREDICTION (LINEAR REGRESSION) ${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo -e "Target Server : ${YELLOW}${REMOTE_USER}@${SERVER_IP}${NC}"
echo -e "Remote Folder : ${YELLOW}${REMOTE_DIR}${NC}"
echo -e "SSH Key       : ${YELLOW}${SSH_KEY}${NC}"
echo ""

# 1. Kiểm tra SSH Key
if [ ! -f "${SSH_KEY}" ]; then
    echo -e "${RED}❌ Không tìm thấy SSH Key tại: ${SSH_KEY}${NC}"
    exit 1
fi

# 2. Kiểm tra kết nối SSH tới máy chủ Colab
echo -e "${BLUE}🔍 [1/5] Kiểm tra kết nối SSH đến Colab server (${SERVER_IP})...${NC}"
if ! ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${REMOTE_USER}@${SERVER_IP}" "echo 'Connected'" > /dev/null 2>&1; then
    echo -e "${RED}❌ Không thể kết nối tới ${REMOTE_USER}@${SERVER_IP}. Hãy kiểm tra Tailscale hoặc bật lại Colab.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Kết nối SSH tới Colab Server thành công!${NC}"

# 3. Chạy CI Pipeline cục bộ (Unit Tests)
echo -e "${BLUE}🧪 [2/5] Thực thi CI Automated Tests (Pytest)...${NC}"
if command -v pytest &> /dev/null; then
    if pytest -v "${PROJECT_DIR}/tests/"; then
        echo -e "${GREEN}✓ Tất cả unit tests đều vượt qua thành công!${NC}"
    else
        echo -e "${RED}❌ Unit tests thất bại! Hủy bỏ quá trình triển khai.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️ Không tìm thấy pytest trong môi trường hiện tại, bỏ qua bước test local.${NC}"
fi

# 4. Đồng bộ mã nguồn lên Colab
echo -e "${BLUE}📦 [3/5] Đồng bộ mã nguồn dự án sang ${REMOTE_DIR}...${NC}"
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${REMOTE_USER}@${SERVER_IP}" "mkdir -p ${REMOTE_DIR}"

rsync -avz -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '.pytest_cache' \
    --exclude '*.log' \
    --exclude '*.pyc' \
    "${PROJECT_DIR}/" "${REMOTE_USER}@${SERVER_IP}:${REMOTE_DIR}/"

echo -e "${GREEN}✓ Đồng bộ mã nguồn hoàn tất!${NC}"

# 5. Kích hoạt và triển khai trên máy chủ
echo -e "${BLUE}⚡ [4/5] Khởi động ứng dụng FastAPI trên Colab Server...${NC}"
ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${REMOTE_USER}@${SERVER_IP}" \
    "chmod +x ${REMOTE_DIR}/run_server.sh && bash ${REMOTE_DIR}/run_server.sh"

# 6. Kiểm tra Health check
echo -e "${BLUE}🩺 [5/5] Kiểm tra sức khỏe dịch vụ (Health Check)...${NC}"
HEALTHY=false
for i in {1..8}; do
    STATUS=$(curl -s -m 3 "http://${SERVER_IP}:8000/health" || echo "failed")
    if echo "${STATUS}" | grep -q "healthy"; then
        HEALTHY=true
        break
    fi
    echo -e "${YELLOW}⏳ Đang chờ dịch vụ sẵn sàng (thử lại ${i}/8)...${NC}"
    sleep 2
done

# 7. Trích xuất Cloudflare Public URL (nếu có)
PUBLIC_URL=$(ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${REMOTE_USER}@${SERVER_IP}" \
    "grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' /tmp/cf_tunnel.log 2>/dev/null | tail -n 1 || echo ''")

echo ""
if [ "$HEALTHY" = true ]; then
    echo -e "${GREEN}${BOLD}🎉 TRIỂN KHAI THÀNH CÔNG 100%! ỨNG DỤNG ĐÃ ONLINE!${NC}"
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
    echo -e "🌐 ${BOLD}Link Nội Bộ (Tailscale Mesh):${NC}  ${YELLOW}http://${SERVER_IP}:8000${NC}"
    if [ -n "$PUBLIC_URL" ]; then
        echo -e "🌍 ${BOLD}Link Công Khai (Internet):${NC}     ${GREEN}${BOLD}${PUBLIC_URL}${NC}"
    fi
    echo -e "📖 ${BOLD}Tài Liệu Swagger API:${NC}          ${YELLOW}http://${SERVER_IP}:8000/docs${NC}"
    echo -e "📜 ${BOLD}Xem Log Trực Tiếp:${NC}             ssh -i ${SSH_KEY} ${REMOTE_USER}@${SERVER_IP} 'tail -f ${REMOTE_DIR}/server.log'"
    echo -e "${CYAN}----------------------------------------------------------------------${NC}"
else
    echo -e "${RED}⚠️ Dịch vụ chưa phản hồi trên cổng 8000. Nhật ký lỗi gần nhất:${NC}"
    ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${REMOTE_USER}@${SERVER_IP}" \
        "tail -n 25 ${REMOTE_DIR}/server.log 2>/dev/null || true"
fi
