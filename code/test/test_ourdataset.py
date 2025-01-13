import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将上级目录添加到模块搜索路径
sys.path.append(parent_dir)
from utils_0 import read_task_in_jsonl, output_jsonl, useful_design_smell
from tqdm import tqdm

import subprocess
import time


test_file = "../../dataset/MLCQ_Qscore_cateory_balance_new_10000_test.jsonl"



def get_free_gpus(min_free=2):
    """使用gpustat检查空闲的GPU数量，返回空闲GPU的ID列表."""
    try:
        # 获取gpustat输出
        gpustat_output = subprocess.check_output("gpustat --json", shell=True).decode("utf-8")
        import json
        gpu_status = json.loads(gpustat_output)
        # print(gpu_status)
        
        # 遍历gpu状态，获取空闲的GPU（内存占用为0）
        free_gpus = [str(i) for i, gpu in enumerate(gpu_status['gpus']) if gpu['memory.used'] < 2000]
        
        # 如果空闲GPU数量达到要求，返回其ID列表
        if len(free_gpus) >= min_free:
            return free_gpus[:min_free]
        else:
            return []
    except Exception as e:
        print(f"Error in checking GPUs: {e}")
        return []
    
from multiprocessing import Process
import time

def start_test(gpu, start, end,i, data_path, model_path):
    """在新的tmux会话中启动训练进程."""
    
    test_command = f"""export CUDA_VISIBLE_DEVICES={gpu}\npython test_detail_2_0.py --start {start} --end {end} --i {i} --data_path {data_path} --model_path {model_path}"""
    
    # 启动tmux会话并执行训练命令
    try:
        # 创建tmux会话并运行训练
        print(test_command)
        subprocess.run(test_command, shell = True)
        # stdout, stderr = process.communicate()
        # print("Output:", stdout.decode())
    except Exception as e:
        print(f"Error in starting tmux session: {e}")
    # return process

def merge_test_jsonl(output_folder):

    x = []
    for i in range(4):
        path = os.path.join(output_folder,str(i)+"_inference.jsonl")
        x += read_task_in_jsonl(path)
    output_jsonl(x, os.path.join(output_folder,"merged_inference.jsonl"))
# 主程序


def multi_label_test():
    model_path = "model/newbase"

    processes = []
    data_path = test_file
    dataset = read_task_in_jsonl(data_path)
    try:
        each_train_min_gpus = 4
     
       
           
        free_gpus = None
        while free_gpus is None or len(free_gpus) < each_train_min_gpus:
            if free_gpus is not None:
                time.sleep(30)
            free_gpus = get_free_gpus(min_free=each_train_min_gpus)
            print(free_gpus)
        
        N = len(dataset) // len(free_gpus)
        print(free_gpus)
        for i, gpu in enumerate(free_gpus):
            start = i * N
            end = (i+1) * N if i + 1 < len(free_gpus) else len(dataset)
            p = Process(target=start_test, args=(gpu, start, end,i, data_path, model_path))
            processes.append(p)
            p.start()  # 启动进程
            time.sleep(30)
            
    except Exception as e:
        print(e)
        for p in processes:
            p.kill()
    for p in processes:
        p.join()
    model_name = model_path.split("/")[-1]
    output_folder = f"../../{model_name}"
    merge_test_jsonl(output_folder)


def main():
    # 检查空闲GPU
    multi_label_test()
    # binary_test()
  
        

if __name__ == "__main__":
    main()