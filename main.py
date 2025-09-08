from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from dotenv import load_dotenv
load_dotenv() 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

from textChatMode.chat import router as ask_router
from LevelDetection.router.levelDetection import router as level_detection_router

app = FastAPI()
 
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes from other files
app.include_router(ask_router)
app.include_router(level_detection_router)

@app.get("/")
def root():
    return {"message": "All endpoints loaded successfully"}
