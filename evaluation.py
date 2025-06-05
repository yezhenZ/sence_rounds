from pytorch_msssim import ssim, ms_ssim
import torch
import numpy as np
#转为png
import cairosvg
from PIL import Image
from torchvision import transforms
import torch
import io
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import lpips
def svg_to_tensor(svg_path,name, size=(224, 224)):
    # 将 SVG 渲染为 PNG 格式（保存在内存中）
    png_bytes = cairosvg.svg2png(url=svg_path, output_width=size[0], output_height=size[1])

    # 从内存加载 PNG 图像
    image = Image.open(io.BytesIO(png_bytes))
    # 创建白色背景图像
    background = Image.new("RGB", image.size, (255, 255, 255))
    # 将透明图像粘贴到背景图上（alpha合成）
    background.paste(image, mask=image.split()[3])  # 使用 alpha 通道作为 mask
    # Step 3: 保存图像为 PNG 文件
    save_path = f"./eval/{name}_all.png"
    background.save(save_path)
    print(f"图像已保存到：{save_path}")
    # 转为 Tensor：[3, H, W]，然后添加 batch 维度变成 [1, 3, H, W]
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 [0,1] float tensor，形状 [3,H,W]
    ])
    image_tensor = transform(background).unsqueeze(0)  # [1, 3, H, W]

    return image_tensor

def load_png_to_tensor(png_path,size=(224, 224)):
    image = Image.open(png_path).convert('RGB')  # 保证是3通道
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # 转为 [0,1] float tensor，形状 [3,H,W]
    ])
    image_tensor = transform(image).unsqueeze(0)  # 加 batch 维度 -> [1, 3, H, W]
    return image_tensor

def evaluate_msssim(img1, img2):
    """
    img1, img2: numpy arrays, shape [H, W, C], 数值范围0~1或0~255，必须一致
    返回 MS-SSIM 值，范围[0,1]，越接近1越相似
    """
    # 确保图像是float且归一化到[0,1]
    if img1.dtype != np.float32 and img1.dtype != np.float64:
        img1 = img1.astype(np.float32) / 255.0
    if img2.dtype != np.float32 and img2.dtype != np.float64:
        img2 = img2.astype(np.float32) / 255.0

    msssim, _ = ssim(img1, img2, multichannel=True, gaussian_weights=True, full=True)
    return msssim

# bull_new = svg_to_tensor("./eval/all_l4_bull_seed0_best.svg", name="bull",size=(224, 224))
# house_new = svg_to_tensor("./eval/all_l4_house_seed0_best.svg", name="house",size=(224, 224))
# bull=load_png_to_tensor("./eval/bull.png")
# print(bull_new)

#
# img = Image.open("./eval/bull.png").convert('RGB')
# # img = img.resize((224, 224))
#
# # 然后再转为 numpy 数组
# img = np.array(img)
#
# img_new = Image.open("./eval/bull_all.png").convert('RGB')
# img_new = img_new.resize((500, 500))
#
# # 然后再转为 numpy 数组
# img_new = np.array(img_new)
#
# score = evaluate_msssim(img, img_new)
# print(f"MS-SSIM: {score:.4f}")


# 初始化 LPIPS 模型，默认使用 AlexNet 特征提取器
lpips_fn = lpips.LPIPS(net='alex')  # 可选: 'alex', 'vgg', 'squeeze'

# img1=svg_to_tensor("./eval/all_l4_bull_seed0_best.svg", name="bull",size=(224, 224))
# img2=svg_to_tensor("./eval/all_l4_house_seed0_best.svg", name="house",size=(224, 224))
img1=load_png_to_tensor("./eval/house_all.png")
target=load_png_to_tensor("./eval/house.png")
print(img1.shape)
# 计算 LPIPS 距离
distance = lpips_fn( target,img1)
print("LPIPS 距离：", distance.item())
