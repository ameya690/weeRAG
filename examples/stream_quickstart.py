# Requires: pip install fastapi uvicorn
# Run: uvicorn wee.stream:app --reload
# Then open: http://127.0.0.1:8000/stream?text=streaming+tokens+yay
print("To try streaming: `uvicorn wee.stream:app --reload` then GET /stream?text=hello`)")
