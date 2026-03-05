"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import init_services, router
from src.data.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    init_services()
    logging.getLogger("app").info("Adaptive Meter Reader started")
    yield
    logging.getLogger("app").info("Shutting down")


app = FastAPI(
    title="Adaptive Meter Reader",
    description="Self-improving meter reading system with multi-turn operator interaction",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
