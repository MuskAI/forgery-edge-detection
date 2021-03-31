"""

"""
from PIL import Image


# 将图片转化为RGB
def make_regalur_image(img, size=(64, 64)):
    gray_image = img.resize(size).convert('RGB')
    return gray_image


# 计算直方图
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    return hist


# 计算相似度
def calc_similar(li, ri):
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    print(calc_sim)
    return calc_sim




if __name__ == '__main__':
    image1 = Image.open('./imageData/img1.png')
    image1 = calc_similar(image1)
    image2 = Image.open('./imageData/img2.jpg')
    image2 = make_regalur_image(image2)
    image3= Image.open('./imageData/img3.png')
    image3 = make_regalur_image(image3)
    print("img1和img2的相似度：", calc_similar(image1, image2))
    print("img1和img3的相似度：", calc_similar(image1, image3))
    print("img2和img3的相似度：", calc_similar(image2, image3))

