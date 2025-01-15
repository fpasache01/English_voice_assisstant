from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel

# Database configuration
DATABASE_URL = "postgresql://myuser:mypassword@localhost:5432/mydatabase"

# Initialize database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the Transcription database model
class TranscriptionDB(Base):
    __tablename__ = "transcriptions"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    transcription = Column(Text)
    response = Column(Text)
# Create the database tables
Base.metadata.create_all(bind=engine)

class TranscriptionInput(BaseModel):
    filename: str
    transcription: str
    response: str


# FastAPI app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Store connected WebSocket clients
clients = []

# Helper function to fetch the last transcription
def get_last_transcription():
    db = SessionLocal()
    try:
        return db.query(TranscriptionDB).order_by(TranscriptionDB.id.desc()).first()
    finally:
        db.close()
# GET /transcriptions/
@app.get("/transcriptions/")
def get_transcriptions():
    """
    Fetch all saved transcriptions.
    """
    db = SessionLocal()
    try:
        transcriptions = db.query(TranscriptionDB).all()
        return [
            {
                "id": t.id,
                "filename": t.filename,
                "transcription": t.transcription,
                "response": t.response,
            }
            for t in transcriptions
        ]
    finally:
        db.close()        

@app.get("/", response_class=HTMLResponse)
async def read_transcriptions(request: Request):
    """
    Renders the main page for viewing transcriptions.
    """
    return templates.TemplateResponse("transcriptions.html", {"request": request})

@app.post("/transcriptions/")
async def create_transcription(data: TranscriptionInput):
    db = SessionLocal()
    try:
        new_transcription = TranscriptionDB(
            filename=data.filename,
            transcription=data.transcription,
            response=data.response,
        )
        db.add(new_transcription)
        db.commit()
        db.refresh(new_transcription)
        return {"id": new_transcription.id, "message": "Transcription added successfully"}
    finally:
        db.close()
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to push real-time updates to clients.
    """
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep the connection open
    except WebSocketDisconnect:
        clients.remove(websocket)
