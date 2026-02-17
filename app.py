from fastapi import FastAPI, UploadFile
from ai_analyzer import ImageAnalytic
from enhancer import SmartEnhancer
import os

app = FastAPI()
enhancer = SmartEnhancer()

@app.post("/process")
async def process_image(file: UploadFile):
    temp_path = f"uploads/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # 1. AI decides what to do
    analyzer = ImageAnalytic(temp_path)
    plan = analyzer.get_action_plan()

    # 2. Execute plan
    result = enhancer.run(temp_path, plan)

    return {"status": "success", "ai_decision": plan, "result_url": result}