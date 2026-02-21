from app.persistence.database import engine
from app.persistence.models import Base

Base.metadata.create_all(bind=engine)
print("Tables created")
