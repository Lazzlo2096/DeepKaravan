import uvicorn

if __name__ == "__main__":
    # One API process owns the IPC queues and one model process.
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, workers=1)
