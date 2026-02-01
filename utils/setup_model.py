import os
import shutil
from pathlib import Path
import sys
import subprocess
#define path

Model_Path =Path("../Triton_Server/model_repository")
Model_Name="yolov8s_detector"
Model_DIR=Model_Path / Model_Name
VERSION_PATH=Model_DIR/"1"
ONNX_FILE=VERSION_PATH/"model.onnx"

def create_dir():
    print("Creating Directory structure...")
    Model_DIR.mkdir(parents=True,exist_ok=True)
    VERSION_PATH.mkdir(parents=True,exist_ok=True)

def Download_export():
    print("Download the model and export into onnx")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not found . installing ultralytics")
        subprocess.check_call([sys.executable,"-m","pip","install","ultralytics"])
    except Exception as e:
        print(f"AN unexcected error occurred : {e}")
    model=YOLO('yolov8s.pt')
    model.export(format="onnx",dynamic=True,simplify=True,opset=12)
    print(" -----> Export complete")

def Move_model():
    print("Move the model into model repository...")
    source_file=Path("yolov8s.onnx")
    if source_file.exists():
        shutil.move(str(source_file),str(ONNX_FILE))
        print(f"  --> Moved to : {ONNX_FILE}")
    else:
        print("please check file ")
        exit(1)

def generate_config():
    print("Generating config.pbtxt")
    config_content="""
    name:"yolov8s_detector"
    platform:"onnxruntime_onnx"
    max_batch_size:8
    input [
    {
    name: "images"
    data_type: TYPE_FP32
    dims:[3,-1,-1]
    }]
    output:[
    {
    name:"output0"
    data_type:TYPE_FP32
    dims:[84,-1]
    }]
    dynamic_batching{
    max_queue_delay_microseconds:100
    }"""
    config_file=Model_DIR/"config.pbtxt"
    with open(config_file,"w") as f:
        f.write(config_content.strip())
    print(f"config file create {config_file}")

if __name__=="__main__":
    create_dir()
    Download_export()
    Move_model()
    generate_config()
    print("\nSUCCESS! Your YOLOv8 model is ready in triton-server/model_repository/yolov8s_detector/")
