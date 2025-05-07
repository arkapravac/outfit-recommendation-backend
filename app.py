from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime
import os
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Fashion Assistant",
            description="AI-powered shopping assistant with personalized recommendations")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
db = None

@app.on_event("startup")
async def startup_db_client():
    global db
    client = AsyncIOMotorClient(MONGO_URL)
    db = client.fashion_assistant

@app.on_event("shutdown")
async def shutdown_db_client():
    if db:
        db.client.close()

@app.get("/")
async def root():
    return {"message": "Welcome to AI Fashion Assistant API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "AI Fashion Assistant API",
        "database_connected": db is not None
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)