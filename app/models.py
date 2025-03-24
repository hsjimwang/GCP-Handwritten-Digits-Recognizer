from sqlalchemy import Column, Float, Integer

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    digit = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
