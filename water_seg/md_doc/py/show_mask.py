import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 定义地物类型的颜色字典
land_cover_colors = {
    1: (200, 0, 0),     # industrial area
    2: (0, 200, 0),     # paddy field
    3: (150, 250, 0),   # irrigated field
    4: (150, 200, 150), # dry cropland
    5: (200, 0, 200),   # garden land
    6: (150, 0, 250),   # arbor forest
    7: (150, 150, 250), # shrub forest
    8: (200, 150, 200), # park
    9: (250, 200, 0),   # natural meadow
    10: (200, 200, 0),  # artificial meadow
    11: (0, 0, 200),    # river
    12: (250, 0, 150),  # urban residential
    13: (0, 150, 200),  # lake
    14: (0, 200, 250),  # pond
    15: (150, 200, 250),# fish pond
    16: (250, 250, 250),# snow
    17: (200, 200, 200),# bareland
    18: (200, 150, 150),# rural residential
    19: (250, 200, 150),# stadium
    20: (150, 150, 0),  # square
    21: (250, 150, 150),# road
    22: (250, 150, 0),  # overpass
    23: (250, 200, 250),# railway station
    24: (200, 150, 0),  # airport
    0: (0, 0, 0)        # unlabeled
}


land_cover_types = {
    1: "industrial area",
    2: "paddyfield",
    3: "irrigated field",
    4: "dry cropland",
    5: "garden land",
    6: "arbor forest",
    7: "shrub forest",
    8: "park",
    9: "natural meadow",
    10: "artificial meadow",
    11: "river",
    12: "urban residential",
    13: "lake",
    14: "pond",
    15: "fish pond",
    16: "snow",
    17: "bareland",
    18: "rural residential",
    19: "stadium",
    20: "square",
    21: "road",
    22: "overpass",
    23: "railway station",
    24: "airport",
    0: "unlabeled"
}



# 定义函数：将mask图像转换为彩色图像
def convert_mask_to_color(image_array, land_cover_colors):
    # 创建一个新的彩色图像
    color_image = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    # 将每个像素值转换为对应的颜色
    for value, color in land_cover_colors.items():
        mask = (image_array == value)
        color_image[mask] = color

    return color_image


# 读取图片
image_path = './caijian/label/GF2_PMS1__L1A0000564539-MSS1_tile_0.png'
image = Image.open(image_path)
image_array = np.array(image)

# 计算每种地物类型的像素数量
unique, counts = np.unique(image_array, return_counts=True)
pixel_count = dict(zip(unique, counts))

# 计算每种地物类型的比例
total_pixels = image_array.size
pixel_percentage = {land_cover_types[key]: (value / total_pixels) * 100 for key, value in pixel_count.items()}

# 输出结果
for land_cover, percentage in pixel_percentage.items():
    print(f'{land_cover}: {percentage:.2f}%')

# 转换mask图像为彩色图像
color_image_array = convert_mask_to_color(image_array, land_cover_colors)

# 保存和显示彩色图像
color_image = Image.fromarray(color_image_array)
color_image.save('colored_mask.png')
color_image.show()

# 饼图颜色对应
labels = list(pixel_percentage.keys())
sizes = list(pixel_percentage.values())
colors = [tuple(np.array(land_cover_colors[key]) / 255) for key in pixel_count.keys()]

# 可视化地物类型分布
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
