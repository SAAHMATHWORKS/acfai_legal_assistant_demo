# app.py - Main entry point for Hugging Face Spaces
from api.main import app

# Hugging Face Spaces will automatically use this 'app' variable
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)  # HF uses port 7860