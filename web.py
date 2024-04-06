from typing import Union
from typing import Optional
import os
import shutil
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import subprocess
import json
from pathlib import Path
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/detect")
async def detect(
    file_name: Optional[str] = None
):
    try:
        temp_dir = f"temp_chunks/{os.path.splitext(file_name)[0]}"
        detect_results = detection_function(f"{temp_dir}/{file_name}")
        shutil.rmtree(temp_dir)
        return JSONResponse(content=detect_results, status_code=200)
    except HTTPException as e:
        print(e)
        return JSONResponse(status_code=500, detail="Error while detect")

@app.post("/merge_chunks")
async def merge_chunks(
    file_name: str = Form(...),
    totalChunks: int = Form(...)
):
    try:
        temp_dir = f"temp_chunks/{os.path.splitext(file_name)[0]}"
        print(temp_dir)
        if all(os.path.exists(f"{temp_dir}/{file_name}.part_{i}") for i in range(1, totalChunks + 1)):
            # 合并所有分片
            with open(f"{temp_dir}/{file_name}", "wb") as main_file:
                for i in range(1, totalChunks + 1):
                    part_location = f"{temp_dir}/{file_name}.part_{i}"
                    print(part_location)
                    with open(part_location, "rb") as part_file:
                        main_file.write(part_file.read())
                    # 删除已合并的分片
                    os.remove(part_location)
            return JSONResponse(content={"FileName":file_name,"message": "Chunk uploaded successfully"}, status_code=200)
    except HTTPException as e:
        print(e)
        return JSONResponse(status_code=500, detail="Error while merge chunk")

        

@app.post("/upload_chunk")
async def upload_chunk(
    file_name: str = Form(...),
    index: int = Form(...),
    totalChunks: int = Form(...),
    chunk: UploadFile = File(...),
):
    try:
        print(file_name)
        # 保存上传的分片到临时文件
        temp_dir = f"temp_chunks/{os.path.splitext(file_name)[0]}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # 将分片写入临时文件夹
        file_location = f"{temp_dir}/{file_name}.part_{index}"
        print(file_location)
        with open(file_location, "wb+") as file_object:
            file_object.write(chunk.file.read())
        return JSONResponse(content={"chunkNumber": index, "message": "Chunk uploaded successfully"}, status_code=200)
        # 检查是否所有分片都上传完了
        # if all(os.path.exists(f"{temp_dir}/{file_name}.part_{i}") for i in range(1, totalChunks + 1)):
        #     # 合并所有分片
        #     with open(f"{temp_dir}/{file_name}", "wb") as main_file:
        #         for i in range(1, totalChunks + 1):
        #             part_location = f"{temp_dir}/{file_name}.part_{i}"
        #             with open(part_location, "rb") as part_file:
        #                 main_file.write(part_file.read())
        #             # 删除已合并的分片
        #             os.remove(part_location)
                
        #     # 此处做进一步处理，比如调用检测函数
        #     detect_results = detection_function(f"{temp_dir}/{file_name}")
            
        #     # 删除临时目录
        #     shutil.rmtree(temp_dir)
        #     return JSONResponse(content=detect_results, status_code=200)
        # else:
        #     return JSONResponse(content={"chunkNumber": i, "message": "Chunk uploaded successfully"}, status_code=200)
    
    except HTTPException as e:
        print(e)
        return JSONResponse(status_code=500, detail="Error while uploading chunk")

def detection_function(filename: str):

    results = run_detection_script(filename)
    return results

def run_detection_script(image_path: str):
        # yolov5 检测脚本的命令
    command = [
        "python", "detect.py",
        "--weights", "weights/best.pt",
        "--source", image_path,
        "--img", "2048",
        "--device", "cpu",
        "--conf-thres", "0.25",
        "--iou-thres", "0.2",
        "--hide-labels",
        "--hide-conf",
        "--save-txt"
    ]
    # 执行命令
    subprocess.run(command, check=True)
    # 检测结果文件夹，改为 yolov5 输出目录
    output_dir = 'runs/detect/exp'

    # 将每个 txt 文件中的结果转换为 JSON
    results_json = []
    labels_dir = Path(output_dir) / 'labels'
    for txt_path in labels_dir.glob('*.txt'):
        with open(txt_path) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                results_json.append({
                    'class': int(parts[0]),
                    # 'confidence': float(parts[5]),
                    'poly': [float(part) for part in parts[1:10]]  # x, y, width, height
                })
    shutil.rmtree(output_dir)
    return results_json

# def run_detection_script(filename: str):
#     # 模拟目标检测函数
#     # 这里应调用你的目标检测模型并返回检测结果
#     # 返回结果格式应根据前端需求而定
#     return {
#         "filename": filename,
#         "detections": [
#             # 假设的检测结果
#             {"object": "tree", "confidence": 0.9, "box": [100, 200, 150, 250]},
#             # 其他检测结果...
#         ]
#     }