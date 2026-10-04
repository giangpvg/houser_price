"""FastAPI Application Entry Point for House Price Prediction AI."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.ml.predictor import HousePricePredictor

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown tasks."""
    print("🚀 [Startup] Khởi tạo mô hình AI Linear Regression...")
    predictor = HousePricePredictor.get_instance()
    print(
        f"✅ [Startup] Mô hình đã sẵn sàng. Trạng thái: {predictor.get_info()['status']}"
    )
    yield
    print("🛑 [Shutdown] Đang tắt ứng dụng...")


app = FastAPI(
    title="🏡 Housing Price Prediction AI",
    description=(
        "Production-grade RESTful API & Interactive UI dự đoán giá nhà "
        "bằng mô hình Linear Regression (Ordinary Least Squares). "
        "Tự động giải thích đóng góp của từng đặc trưng (Feature Contributions) "
        "và hỗ trợ triển khai CI/CD lên Google Colab Server."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount static directory if exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve interactive web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "service": "Housing Price Prediction AI",
        "status": "online",
        "docs": "/docs",
    }
