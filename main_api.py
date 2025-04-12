from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import Chatbot
import uvicorn

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
bot = Chatbot(config_path="config.ini", test_mode=True)

# Define input schema
class QueryInput(BaseModel):
    input: str

# Define the /query endpoint
@app.post("/query")
def handle_query(query: QueryInput):
    result = bot.run_single_query(query.input)
    return JSONResponse(
        content=result,
        ensure_ascii=False,     # ✅ allow proper emoji/unicode characters
        media_type="application/json; charset=utf-8"  # ✅ explicitly set UTF-8
    )

# Run the server when executing directly
if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000)


