#!/bin/bash
# start.sh

# Start the FastAPI app with production settings
uvicorn main_api:app --host 0.0.0.0 --port 10000
