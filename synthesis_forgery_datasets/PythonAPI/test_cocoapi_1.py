from pycocotools.coco import COCO

val_info = r'D:\实验室\图像篡改检测\数据集\COCO\annotations\annotations_trainval2017\annotations\instances_val2017.json'
val_image = r'D:\实验室\图像篡改检测\数据集\COCO\val2017'

coco = COCO(val_info)  # 导入验证集
all_ids = coco.imgs.keys()
print(len(all_ids))
person_id = coco.getCatIds(catNms=['person'])
print(person_id)
person_imgs_id = coco.getImgIds(catIds=person_id)
print(len(person_imgs_id))
###
'''
loading annotations into memory...
Done (t=1.45s)
creating index...
index created!
5000  # 验证集样本总数
[1]  # 人这个类的类别id
2693  # 在验证集中，包含人这个类的图像有2693张
'''
###

