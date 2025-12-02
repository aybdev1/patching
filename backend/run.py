# run.py
#import uvicorn
#if __name__ == "__main__":
    #uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    #uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# run.py

import os
import uvicorn

if __name__ == "__main__":
    # Safe default: no reload in production
    reload_mode = os.getenv("UVICORN_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("APP_PORT", 8000)),
        reload=reload_mode
    )


 
    
