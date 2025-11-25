from fastapi import FastAPI
import httpx
import subprocess
import os

import tempfile
import shutil

app = FastAPI()

@app.post("/convert")
def convert(payload: dict):
    
    input_path = payload["npy"]
    save_name = f'{input_path.split("/")[-1].split(".")[0]}'

    avatar_id = payload["avatar"]
    
    audio_path = payload["audio"]

    save_path = "/workspace/ARTalk_speech-to-expression/"
    # 새로운 환경변수 만들기
    env = os.environ.copy()
    env["PYTHONPATH"] = f'{os.path.realpath("./")}:{env.get("PYTHONPATH", "")}'

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = tmpdir
        subprocess.run(["python", "scripts/convert/convert_expression.py",
                        "--input_path", input_path, "--output_path", f"{output_path}/{save_name}"],
                        env=env)
        
        subprocess.run(["python", "gaussianavatars/animate.py",
                        "--model_path", f"examples/output/{avatar_id}/avatar/",
                        "--target_animation_path", f"{output_path}/{save_name}/fit.npz",
                        "--target_cam_trajectory_path", f"{output_path}/{save_name}/cam_static.npz",
                        "--output_path", output_path,
                        "--audio_path", audio_path],
                        env=env) 
        

        shutil.copy(os.path.join(tmpdir, "renders.mp4"), save_path)

    
    video_path = os.path.join(save_path, "renders.mp4")
    return {"status": "done", "video": video_path}
