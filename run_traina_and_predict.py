import subprocess
import os
import argparse

def run_train(cuda_devices, train_json):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["WANDB_DISABLED"] = "true"
    result = subprocess.run([
        "/root/anaconda3/envs/moelora/bin/python", 
        "/root/lhy/MOELoRA-peft/run_mlora.py", 
        train_json
    ], capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception("Training process failed")

def run_predict(cuda_devices, predict_json):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["WANDB_DISABLED"] = "true"
    result = subprocess.run([
        "/root/anaconda3/envs/moelora/bin/python", 
        "/root/lhy/MOELoRA-peft/run_mlora.py", 
        predict_json
    ], capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception("Prediction process failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run train and predict with specified CUDA devices.")
    parser.add_argument('--cuda_devices', type=str, default="0", help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--train_json', type=str, required=True, help='Path to the train JSON file')
    parser.add_argument('--predict_json', type=str, required=True, help='Path to the predict JSON file')
    
    args = parser.parse_args()
    cuda_devices = args.cuda_devices
    train_json = args.train_json
    predict_json = args.predict_json

    try:
        run_train(cuda_devices, train_json)
        run_predict(cuda_devices, predict_json)
    except Exception as e:
        print(f"Error: {e}")
