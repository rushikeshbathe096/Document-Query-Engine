import pytest
import app.persistence.database as database
from app.persistence.database import Base


@pytest.fixture(autouse=True)
def setup_test_db():
    # 🔥 Reset DB before every test
    Base.metadata.drop_all(bind=database.engine)
    Base.metadata.create_all(bind=database.engine)

    yield