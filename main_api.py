from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import Chatbot
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the chatbot
bot = Chatbot(config_path="config.ini", test_mode=True)

# Define input schema
class QueryInput(BaseModel):
    input: str

# Define the /query endpoint
@app.post("/query")
def handle_query(query: QueryInput):
    result = bot.run_single_query(query.input)
    return {"response": result}

# Run the server when executing directly
if __name__ == "__main__":
    uvicorn.run("main_api:app", host="0.0.0.0", port=8000)

