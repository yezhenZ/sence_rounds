import subprocess as sp
import argparse
import time

# ===================================================
# ================= run all ==================
# ===================================================
# This script runs both foreground and background one after the other
# If you have a single GPU, you should use this script.
# Example of running commands:
# CUDA_VISIBLE_DEVICES=6 python scripts/run_all.py --im_name "man_flowers"
# CUDA_VISIBLE_DEVICES=2 python scripts/run_all.py --im_name "hummingbird"
# CUDA_VISIBLE_DEVICES=3 python scripts/run_all.py --im_name "boat"
# ===================================================

# list of divs per layer
#默认4

parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer", type=str, default='4')
parser.add_argument("--num_strokes", type=int, default=32)

args = parser.parse_args()

# run the first row (fidelity axis)
start_time_fidelity_b = time.time()
#测试默认一层layer=4，默认只有一个seed0

for l in args.layer:
    # if not os.path.exists(f"./results_sketches/{args.im_name}/runs/background_l{l}_{args.im_name}_mask/points_mlp.pt"):
    #背景
    sp.run(["python", "scripts/generate_fidelity_levels.py",
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background_or_all", "background",
           "--num_strokes",str(args.num_strokes)])
    end_time_fidelity_b = time.time() - start_time_fidelity_b
    print("=" * 50)
    print(f"end_time_background {str(l)} [{end_time_fidelity_b:.3f}]")



    #前景对象
    start_time_fidelity_b = time.time()
    sp.run(["python", "scripts/generate_fidelity_levels.py",
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background_or_all", "object",
            "--num_strokes", str(args.num_strokes)])
    end_time_fidelity_b = time.time() - start_time_fidelity_b
    print("=" * 50)
    print(f"end_time_object {str(l)} [{end_time_fidelity_b:.3f}]")
    

    #合并
    start_time_fidelity_b = time.time()
    sp.run(["python", "combine.py",
            "--im_name", args.im_name])
    end_time_fidelity_b = time.time() - start_time_fidelity_b
    print("=" * 50)
    print(f"combine {str(l)} [{end_time_fidelity_b:.3f}]")


    #第二轮边缘优化
    start_time_fidelity_b = time.time()
    sp.run(["python", "scripts/generate_fidelity_levels.py",
            "--im_name", args.im_name,
            "--layer_opt", str(l),
            "--object_or_background_or_all", "all",
            "--num_strokes", str(args.num_strokes)])
    end_time_fidelity_b = time.time() - start_time_fidelity_b
    print("=" * 50)
    print(f"end_time_all {str(l)} [{end_time_fidelity_b:.3f}]")



