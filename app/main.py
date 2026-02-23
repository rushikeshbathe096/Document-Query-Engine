from fastapi import FastAPI

from app.api.routes import router as api_router
from app.persistence.database import Base, engine


def create_app() -> FastAPI:
    app = FastAPI(
        title="Intelligent Document Query System",
        description="Evidence-driven document intelligence backend",
        version="1.0.0",
    )

    # Initialize database tables
    Base.metadata.create_all(bind=engine)

    # Register API routes
    app.include_router(api_router)

    return app


app = create_app()
