from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import Chatbot
import uvicorn
from fastapi.responses import JSONResponse
from typing import Optional

# Initialize FastAPI app
app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set your frontend domain for better security if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load the chatbot using improved unified logic
bot = Chatbot(config_path="config.ini", test_mode=False)

# 📬 Input schema for /query
class QueryInput(BaseModel):
    input: str

# 📬 Input schema for /feedback
class FeedbackInput(BaseModel):
    input: str
    was_helpful: bool
    predicted: str
    corrected: Optional[str] = None

# 🔁 Chat query handler (now uses unified logic)
@app.post("/query")
def handle_query(query: QueryInput):
    result = bot.handle_query(query.input)  # ✅ uses improved shared logic
    return JSONResponse(content=result, media_type="application/json; charset=utf-8")

# ✅ Feedback handler
@app.post("/feedback")
def log_user_feedback(feedback: FeedbackInput):
    bot.feedback_logger.log_feedback(
        user_input=feedback.input,
        predicted_emergency=feedback.predicted,
        was_helpful=feedback.was_helpful,
        confirmed_emergency=feedback.corrected
    )
    return {"message": "Feedback saved"}

# 🏁 Run locally if executed directly
if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000)



