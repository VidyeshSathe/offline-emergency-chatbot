from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import Chatbot
import uvicorn
from fastapi.responses import JSONResponse
from fastapi import Body
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can change "*" to your frontend domain for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the chatbot
bot = Chatbot(config_path="config.ini", test_mode=False)

# Define input schema
class QueryInput(BaseModel):
    input: str

# Define the /query endpoint
@app.post("/query")
def handle_query(query: QueryInput):
    result = bot.run_single_query(query.input)
    return JSONResponse(content=result, media_type="application/json; charset=utf-8")
# Feedback input model
class FeedbackInput(BaseModel):
    input: str
    was_helpful: bool
    predicted: str
    corrected: Optional[str] = None

# Define /feedback endpoint
@app.post("/feedback")
def log_user_feedback(feedback: FeedbackInput):
    bot.feedback_logger.log_feedback(
        user_input=feedback.input,
        predicted_emergency=feedback.predicted,
        was_helpful=feedback.was_helpful,
        confirmed_emergency=feedback.corrected
    )
    return {"message": "Feedback saved"}

# Run the server when executing directly
if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000)




