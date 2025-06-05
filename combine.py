import subprocess as sp
import argparse
import time
import torch
import os
import pydiffvg
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
from shutil import copyfile
import numpy as np
import imageio
from skimage.transform import resize
from models.painter_params import MLP, WidthMLP
import sketch_utils
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import random
import svgwrite
#复制best.及对应参数
def copy_files(folders_arr, back_or_obj, output_subdir):
    new_filename = None
    for j, folder_ in enumerate(folders_arr):
        print(f'{folders_arr}')
        cur_f = f"{runs_dir}/{folder_}"
        if os.path.exists(cur_f) and  not folder_.startswith("all"):
            #找best
            svg_filename_lst = [s_ for s_ in os.listdir(cur_f) if s_.endswith("best.svg")]  # [0]
            #找points
            pt_filename_lst = [s_ for s_ in os.listdir(cur_f) if s_ == "points_mlp.pt"]
            #找init
            init_lst = [s_ for s_ in os.listdir(cur_f) if s_ == "init.svg"]
            

            copyfile(f"{cur_f}/{svg_filename_lst[0]}", f"{output_subdir}/ best_{back_or_obj}.svg")
            copyfile(f"{cur_f}/{pt_filename_lst[0]}", f"{output_subdir}/points_mlp_{back_or_obj}.pt")
            copyfile(f"{cur_f}/{init_lst[0]}", f"{output_subdir}/init_{back_or_obj}.svg")
            if back_or_obj == "obj":
                mask = [s_ for s_ in os.listdir(cur_f) if s_ == "mask.png"]
                #这里注意我只写了seed0的情况
                cur_f_seed0= [s_ for s_ in os.listdir(cur_f) if s_.endswith("seed0")]

                unmasked_input =[s_ for s_ in os.listdir(f"{cur_f}/{cur_f_seed0[0]}") if s_.endswith("input.png")]
                print(f"the seed o is {unmasked_input}")
                copyfile(f"{cur_f}/{cur_f_seed0[0]}/{unmasked_input[0]}", f"{output_subdir}/{unmasked_input[0]}")

                copyfile(f"{cur_f}/{mask[0]}", f"{output_subdir}/{mask[0]}")

        else:
            print(f"{folder_} find best mask points init failed!! ")


def combine_bo(combine_dir, im_name, layers):
    runs_folders = os.listdir(runs_dir)
    for layer in layers:
        #默认没有ratio，有要删除！
        object_paths = [path for path in runs_folders if "mask" not in path]
        background_paths = [path for path in runs_folders if "mask" in path]
        # print(object_paths)
        # print(layer_paths_o)
        copy_files(object_paths, "obj", f"{combine_dir}/obj")
        copy_files(background_paths, "back", f"{combine_dir}/back")

    # # 加载模型参数
    # state_dict = torch.load(f"{combine_dir}/points_mlp_obj.pt", map_location='cpu')  # 加上 map_location 方便在 CPU 查看
    # # 查看模型参数名
    # print("Model state dict keys:")
    # for key ,value in state_dict['model_state_dict'].items():
    #     print(f"{key}: {value.shape}")
    # # 查看优化器参数名
    # print("\nOptimizer state dict keys:")
    # for key in state_dict['optimizer_state_dict'].keys():
    #     print(key)


def get_combined_points(args, combine_dir, back_or_obj, eps=1e-4):
    img_dir = f"{combine_dir}/{back_or_obj}"
    mlp_points_weights_path = f"{img_dir}/points_mlp_{back_or_obj}.pt"
    sketch_init_path = f"{img_dir}/init_{back_or_obj}.svg"
    device = args.device
    control_points_per_seg = args.control_points_per_seg
    num_paths = args.num_paths
    mlp = MLP(num_strokes=num_paths, num_cp=control_points_per_seg).to(device)
    checkpoint = torch.load(mlp_points_weights_path)
    mlp.load_state_dict(checkpoint['model_state_dict'])

    points_vars, canvas_width, canvas_height = sketch_utils.get_init_points(sketch_init_path)
    points_vars = torch.stack(points_vars).unsqueeze(0).to(device)
    points_vars = points_vars / canvas_width
    points_vars = 2 * points_vars - 1
    points = mlp(points_vars)
    points = 0.5 * (points + 1.0) * canvas_width
    points = points + eps * torch.randn_like(points)
    return points, canvas_width, canvas_height


def get_sketch(args, combine_dir, points_obj, points_back_removed, edge_points,canvas_width, canvas_height):
    #points_obj...都是1，nums
    points1 = torch.cat((points_obj, points_back_removed), dim=1)
    #绘制初始笔画
    points_edge=init_new_controlpoints(args,edge_points,canvas_width,canvas_height)
    # # print(points_edge)
    points_vars = torch.cat((points1, points_edge), dim=1)
    output_path = f"{combine_dir}/"

    num_paths_combine = points_vars.shape[1] // (4 * 2)
    width_ = 1.5
    device = args.device
    control_points_per_seg = args.control_points_per_seg
    num_control_points = torch.zeros(1, dtype=torch.int32) + (control_points_per_seg - 2)

    all_points = points_vars.reshape((-1, num_paths_combine, control_points_per_seg, 2))

    shapes = []
    shape_groups = []
    for p in range(num_paths_combine):
        width = torch.tensor(width_)
        w = width / 1.5
        path = pydiffvg.Path(
            num_control_points=num_control_points, points=all_points[:, p].reshape((-1, 2)),
            stroke_width=width, is_closed=False)

        # is_in_canvas_ = sketch_utils.is_in_canvas(canvas_width, canvas_height, path, device)
        # if is_in_canvas_ and w > 0.7:
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=None,
            stroke_color=torch.tensor([0, 0, 0, 1]))
        shape_groups.append(path_group)
    pydiffvg.save_svg(f"{output_path}/best_iter.svg", canvas_width, canvas_height, shapes, shape_groups)


def remove_points(points, combine_dir, canvas_width, canvas_height):
    mask = f"{combine_dir}/obj/mask.png"
    # mask=f"/home/dell/zyl/scene_2/results_sketches/house/object_matrix/mask.png"
    mask = imageio.imread(f'{mask}')  # shape: (H, W)
    # unique_vals = np.unique(mask)
    mask = resize(mask, (canvas_width, canvas_width), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    #前提三通道一样
    if mask.ndim == 3:
        mask = mask[..., 0]  # 通常取红色通道即可

    H, W = mask.shape
    points = points.reshape((-1, 4, 2))
    points_mask = points.clone().long()
    N = points.shape[0]
    print(points.shape)
    #N条线段
    keep_mask = torch.ones(N, dtype=torch.bool)

#可能有问题看看
    for i in range(N):
        for j in range(4):  # 4 control points
            x, y = points_mask[i, j]
            if 0 <= x < W and 0 <= y < H:
                if mask[y, x] == 1:
                    keep_mask[i] = False
                    break

    # 保留没有被 mask 掉的线条
    filtered_points = points[keep_mask]
    filtered_points = filtered_points.reshape((1, -1))
    return filtered_points

#FPS采样算法
def fps(points, num_points):
    """
    FPS采样算法实现
    Args:
        points: 点集，N x 2的二维数组，每行表示一个点的坐标
        num_points: 采样数量
    Returns:
        采样点的索引列表
    """
    n = len(points)
    distances = np.full(n, np.inf)  # 初始化每个点到已采样点集的最短距离为无穷大
    # distances = 0  # 初始化每个点到已采样点集的距离为0
    samples = []  # 采样点索引集
    samples_points = []  # 采样点集
    # 创建一个随机数生成器，使用默认的种子值
    rng = np.random.RandomState()
    current = rng.randint(n)  # 随机选择一个起始点索引
    samples.append(current)  # 将起始点索引加入采样点索引集
    samples_points.append(points[current])  # 将起始点加入采样点集

    while len(samples) < num_points:
        # 计算每个点到已选点集的最短距离
        for i in range(n):
            if i in samples:
                distances[i] = 0
            else:
                dist = np.linalg.norm(points[i] - points[current])
                distances[i] = min(dist, distances[i])
        # 找到距离已选点集最远的点，将它添加到采样点集中
        farthest = np.argmax(distances)#np.argmax() 是NumPy库中的一个函数，用于返回数组中最大元素的索引
        samples.append(farthest)
        samples_points.append(points[farthest])
        current = farthest
        distances[current] = np.inf  # 将新选点的最短距离设为无穷大

    return samples_points

def plot_edge(combine_dir, samples_points):
    # edge_points=samples_points.view(-1, 2).clone().detach().cpu().numpy()
    edge_points = np.array(samples_points.copy())  # 这一步很关键
    #
    dwg = svgwrite.Drawing(f'{combine_dir}/output.svg', size=(224, 224), profile='tiny')
    for x, y in edge_points:
        x = float(x)
        y = float(y)
        dwg.add(dwg.circle(center=(y, x), r=1, fill='black'))
    # 保存 SVG 文件
    dwg.save()
    print("SVG 保存成功：output.svg")
    # currently supports one image (and not a batch)
    plt.figure(figsize=(4, 4))
    plt.subplot()
    # 读取 PNG 图像
    img = Image.open(f"{combine_dir}/obj/input.png")
    # 定义 transform（比如缩放成固定大小、转 tensor）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 可按需设置尺寸
        transforms.ToTensor()  # 转成 [C, H, W]，值域为 [0, 1]
    ])
    # 应用 transform
    img_tensor = transform(img)  # shape: [3, H, W]
    img_tensor = img_tensor.unsqueeze(0)  # shape: [1, 3, H, W] -> 加 batch 维

    print(img_tensor.shape)
    inputs = make_grid(img_tensor, normalize=True, pad_value=0)
    inputs = np.transpose(inputs.cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(inputs, interpolation='nearest', vmin=0, vmax=1)
    if len(edge_points) != 0:
        plt.scatter(edge_points[:, 1], edge_points[:, 0], s=40, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{combine_dir}/edge_samples.png")
    plt.close()


#在边缘采样
def Sampling_edge(args,combine_dir, num_edge_points,canvas_width,canvas_height):
    region_points = []
    # Read image and transfer to gray
    mask_path = f"{combine_dir}/obj/mask.png"
    mask = cv2.imread(mask_path)
    resized_mask = cv2.resize(mask, (canvas_width, canvas_height), interpolation=cv2.INTER_NEAREST)
    # print(resized_mask[resized_mask>0])

    edges = cv2.Canny(resized_mask, 20, 200)
    edge_points = np.argwhere(edges > 0)  # Detect points in edge
    # print(edge_points.shape)
    if edge_points.size == 0:
        return []
    cv2.imwrite(f"{combine_dir}/obj/extrate_edges.png", edges)
    if num_edge_points== 0:
        selected_points = []
    else:
        selected_points = fps(edge_points, num_edge_points)

        # print(len(selected_points))
    # region_points.append(np.array(selected_points))

    # print("region_points",region_points)
    return selected_points

def init_new_controlpoints(args,edge_points,canvas_width,canvas_height):
    points = []

    #先预处理，图像采样的点xy相反
    edge_points = np.array(edge_points).astype(np.float32)
    edge_points_normalised=edge_points.copy()
    edge_points_normalised[:, 0] = edge_points[:, 1] / canvas_width
    edge_points_normalised[:, 1] = edge_points[:, 0] /canvas_width

    for j in range(len(edge_points_normalised)):
        radius = 0.05
        p0 = edge_points_normalised[j]
        points.append([p0[0] * canvas_width, p0[1] * canvas_height])

        for k in range(args.control_points_per_seg - 1):
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            points.append([p1[0] * canvas_width, p1[1] * canvas_height])
            p0 = p1
            # print(f"the p0 is {p0}")
            # print(f"the p1 is {p1}")

    points = torch.tensor(points).to(args.device)
    points = points.view(1, -1)
    return points
    
if __name__ == "__main__":
    #need:back.svg back_init,svg ;obj.svg  obj_init.svg; points_mlpback,  points_mlpobj,mask

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_paths", type=int,
                        default=64, help="number of strokes")
    parser.add_argument("--im_name", type=str, default="")
    parser.add_argument("--control_points_per_seg", type=int, default=4)

    # 改为2411
    parser.add_argument("--layers", type=str, default="4")
    args = parser.parse_args()
    args.device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    layers = args.layers.split(",")

    output_dir = f"./results_sketches/{args.im_name}"

    if not os.path.exists(f"{output_dir}/combine"):
        os.mkdir(f"{output_dir}/combine")
    combine_dir = f"{output_dir}/combine"
    if not os.path.exists(f"{combine_dir}/obj"):
        os.mkdir(f"{combine_dir}/obj")
    if not os.path.exists(f"{combine_dir}/back"):
        os.mkdir(f"{combine_dir}/back")
    device = torch.device("cuda" if (
            torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    runs_dir = f"{output_dir}/runs"

    # 加载两个图并合并
    combine_bo(combine_dir, args.im_name, layers)
    # 获取back and obj 的best_itr.svg
    points_back, canvas_width, canvas_height = get_combined_points(args, combine_dir, back_or_obj="back")
    points_obj, canvas_width, canvas_height = get_combined_points(args, combine_dir, back_or_obj="obj")
    # print(points_back.shape)
    # 获得mask之后的back points
    points_back_removed = remove_points(points_back, combine_dir, canvas_width, canvas_height)
    print(points_back_removed.shape)
    # 采样边缘点points_back:1,sum_points
    num_edge_points = (points_back.shape[1] - points_back_removed.shape[1]) // (4 * 2)
    edge_points = Sampling_edge(args, combine_dir, num_edge_points,canvas_height,canvas_height)

    #画出edge_points情况
    plot_edge(combine_dir,edge_points)
    # 获取最终的sketch
    get_sketch(args, combine_dir, points_obj, points_back_removed,edge_points, canvas_width, canvas_height)
# gen_matrix(output_dir, args.im_name, layers, cols)


# svg_path = f"{output_dir}/background_matrix"
# resize_ob_j =0
# plot_matrix_svg(svg_path, range(9), cols, resize_obj, output_dir, "background_all")
# plot_matrix_svg(svg_path, range(9)[1::2], cols, resize_obj, output_dir, "background_4x4")
#
#
# svg_path = f"{output_dir}/object_matrix"
# resize_ob j =1
# plot_matrix_svg(svg_path, range(9), cols, resize_obj, output_dir, "obj_all")
# plot_matrix_svg(svg_path, range(9)[1::2], cols, resize_obj, output_dir, "obj_4x4")


# plot_matrix_raster(f"{output_dir}/combined_matrix", range(9)[1::2], cols, output_dir, "combined_4x4")
# plot_matrix_raster(f"{output_dir}/combined_matrix", rows, cols, output_dir, "combined_all")
