"""FastAPI 应用装配。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import BASE_DIR, settings
from .api.routes import router as api_router

logger = logging.getLogger("sentiment")

STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动日志。模型是**懒加载**的，这里不读模型。"""
    model_path = settings.model_path
    logger.info(
        "启动 中文情感分析平台 v%s | 模型产物 %s (存在=%s)",
        __version__,
        model_path,
        Path(model_path).exists(),
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="中文情感分析平台",
        description=(
            "中文外卖评论情感二分类。Track A 为纯 numpy/scikit-learn 字符 n-gram 基线，"
            "Track B 为 LoRA 微调管线。返回特征贡献，可解释。"
        ),
        version=__version__,
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.include_router(api_router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api", include_in_schema=False)
    def api_root() -> dict:
        return {
            "name": "中文情感分析平台",
            "version": __version__,
            "docs": "/docs",
            "endpoints": [
                "/api/health",
                "/api/info",
                "/api/predict",
                "/api/predict/batch",
                "/api/metrics",
                "/api/lexicon",
                "/api/models/catalog",
            ],
        }

    return app


app = create_app()