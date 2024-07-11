import cv2

# 读取tif图片
img = cv2.imread("img/land/2.tif", cv2.IMREAD_UNCHANGED)

# 限定大小为1024*1024
resized_img = cv2.resize(img, (1024, 1024))

# 保存为png格式
cv2.imwrite('output_image.png', resized_img)

print('Image compressed and resized successfully.')