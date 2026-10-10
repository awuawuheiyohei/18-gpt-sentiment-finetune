"""开发入口：``python main.py`` → http://127.0.0.1:5055"""
from __future__ import annotations

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5055")),
        reload=bool(os.environ.get("RELOAD")),
    )
