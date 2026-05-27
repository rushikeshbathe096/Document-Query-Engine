import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="document_query_test_"))
TEST_DB_PATH = TEST_DB_DIR / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
import pytest
import app.persistence.database as database
from app.persistence.database import Base


@pytest.fixture(autouse=True)
def setup_test_db():
    # 🔥 Reset DB before every test
    Base.metadata.drop_all(bind=database.engine)
    Base.metadata.create_all(bind=database.engine)

    yield
