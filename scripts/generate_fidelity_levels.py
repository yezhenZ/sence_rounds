import os
import argparse
import subprocess as sp
from shutil import copyfile
import time

# ===========================================
# ======= fidelity axis generation ==========
# ===========================================
# This script is to run the baseline sketching method with the ViT clip model.
# We use this to create the first row in the abstraciton matrix.
# You can use this to create both the objects and background. 
# The default parameters are set for the background case.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=1 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 7 --object_or_background "object" --resize_obj 1
# CUDA_VISIBLE_DEVICES=3 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 7 --object_or_background "background"
# CUDA_VISIBLE_DEVICES=2 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 8 --object_or_background "background"


parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--object_or_background_or_all", type=str, default="background")
parser.add_argument("--resize_obj", type=int, default=0)
parser.add_argument("--num_iter", type=int, default=300)
parser.add_argument("--path_svg", type=str, default="none")
parser.add_argument("--num_strokes", type=int, default=64,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
args = parser.parse_args()


path_to_input_images = "./target_images" # where the input images are located
output_pref = f"./results_sketches/{args.im_name}/runs"
combine_path=f"./results_sketches/{args.im_name}/combine"

# if you run on objects, this need to be changed:
im_filename = f"{args.im_name}_mask.png"
folder_ = "background"
gradnorm = 0
mask_object = 0
num_strokes = 32
if args.object_or_background_or_all == "object":
    if args.layer_opt != 4:
        gradnorm = 1
    mask_object = 1
    im_filename = f"{args.im_name}.png"
    folder_ = "scene"
if args.object_or_background_or_all == "all":
    #其实应该是先combine再有best_iter.svg
    path_svg = f"{combine_path}/best_iter.svg"
    im_filename = f"{args.im_name}.png"
    num_strokes = num_strokes*2
    folder_ = "scene"
else:
    path_svg = "none"
    #采样点



# ===================
# ====== demo =======
# ===================

#xiugai
num_sketches = 1
num_iter = args.num_iter
# ===================

# set the weights for each layer
clip_conv_layer_weights_int = [0 for k in range(12)]
if args.object_or_background_or_all == "object":
    # we combine two layers if we train on objects
    clip_conv_layer_weights_int[4] = 0.5
clip_conv_layer_weights_int[args.layer_opt] = 1
clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

file_ = f"{path_to_input_images}/{folder_}/{im_filename}"
test_name = f"{args.object_or_background_or_all}_l{args.layer_opt}_{os.path.splitext(im_filename)[0]}"
print(f"now is  {args.object_or_background_or_all}")

start_time = time.time()
sp.run(["python", 
        "scripts/run_sketch.py", 
        "--target_file", file_,
        "--output_pref", output_pref,
        "--num_iter", str(num_iter),
        "--test_name", test_name,
        "--num_sketches", str(num_sketches),
        "--mask_object", str(mask_object),
        "--clip_conv_layer_weights", clip_conv_layer_weights,
        "--gradnorm", str(gradnorm),
        "--resize_obj", str(args.resize_obj),
        "--eval_interval", str(50),
        "--path_svg", path_svg,
        "--num_strokes", str(num_strokes),
        "--object_or_background_or_all",str(args.object_or_background_or_all),
        "--im_name",args.im_name,
        "--min_eval_iter", str(400)])
total_time = time.time() - start_time
print(f"Time for one sketch [{total_time:.3f}] seconds")
