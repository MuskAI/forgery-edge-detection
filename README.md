
# forgery-edge-detection
大二到大三时期随刘春晓老师实验室锻炼图像篡改边缘检测项目,forgery-edge-detection下有两个文件夹,这是所有关于数据集生成和代码训练的代码




- [forgery-edge-detection](#forgery-edge-detection)
  * [make_tamper_dataset_from_coo](#make-tamper-dataset-from-coo)
    + [camouflage_dataset_gen(COD10K)](#camouflage-dataset-gen-cod10k-)
    + [for_compare_experiment](#for-compare-experiment)
    + [new_start](#new-start)
    + [PythonAPI](#pythonapi)
    + [to_hey](#to-hey)
    + [tools](#tools)
  * [Mymodel](#mymodel)
    + [datasets](#datasets)
    + [model](#model)
    + [runs](#runs)
    + [save_model](#save-model)
    + [test_tools](#test-tools)
    + [train](#train)
    + [utils](#utils)
  * [functions.py](#functionspy)
  * [utils.py](#utilspy)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>



## make_tamper_dataset_from_coo
这是制作数据集的代码，下面我来一一介绍每个文件
### camouflage_dataset_gen(COD10K)
这是使用伪装检测数据集C0D10K作为原材料制作相似性纹理的篡改数据
<http://dpfan.net/cod10k/>
### for_compare_experiment
1. deal_with_columbia_dataset.py:处理Columbia数据集用来进行指标计算，对gt进行操作。**但是要注意，这个数据集的gt不准确，具体可以看我的技术文档B**
这是模型训练的代码

2. function.py:指标计算函数

3. public_dataset_issues.py 处理casia coverage数据集用来进行指标计算

4. test_helper.py 没在使用了

5. tolerance_f1.py:由于别人是区域，我们是边缘，这里是将别人的区域转化为边缘再进行指标计算

6. value_f1_acc_recall_precision.py 没有再使用了

### new_start
除了COD10K外所有数据集生成代码都在这里，有
1. 利用COCO 生成 splicing 和 copy-move类型的数据
2. 利用template生成
3. 纹理splicing数据，无周期性错位纹理数据

具体的:
1. analysis.py
我拿到casia之后用这个来知道数据集尺寸是怎么样的

2. blur_method.py
考虑全局模糊和局部模糊

3. gen_8_map.py
通过std_gt(标准gt)生成8张图的方法，代码不是很长，看代码就能懂

4. gen_cm_tamper_dataset_1225.py
生成无周期性错位纹理数据

5. gen_cp_tamper_dataset.py 和 gen_cp_tamper_dataset_v2.py
用coco来生成篡改数据，其中有考虑poisson image editing的方法，但是这种方法实际效果并不明显，而且生成速度很慢

6. gen_from_public_dataset.py
一开始我想用casia数据集来训练，但是后面的方案是不使用公开数据训练，所以这部分代码目前也没有使用了

7. gen_negative_dataset.py
生成没有篡改的数据

8. gen_tamper_dataset.py
与5 类似，只有稍微的不同，只需要看懂一个就可以都搞定了

9. gen_texture_tamper_dataset.py
使用纹理数据来生成splicing类型的篡改数据

10. gen_tp_dataset_from_multi_template.py
使用casia的template来生成数据

11. get_double_edge.py
获取双边缘

12. mask_to_fix_background.py
如其名，这是还是2020年7月份写的代码，现在可以不用使用了

13. myCallImage.py
不需要使用

14. poisson_image_editing.py

泊松图像编辑的代码，作为一个备选来使用，初学者可以后面再使用

15. ui.py 
未完成的qt 软件，有兴趣的话可以将这个封装成一个软件

### PythonAPI
使用COCO数据需要用到的API，从COCO数据集官网下载，你需要先去网上了解COCO数据集API的使用，在这里我不做解释

### to_hey
黄恩怡辅助我的时候，给她的代码，如有疑问可以去问黄恩怡

### tools
在过程中需要使用的小工具，如其名，我的代码中凡是以temp命名的都可以直接删掉，稍微注意一下其中的stex_issues.py
这是用来转化stex(我们使用到的纹理数据)的代码，其他的看名字就能懂

## Mymodel

这是模型训练的代码，下面我来一一介绍
### datasets
这是关于数据处理与准备的代码
1. blur_method.py
这个是关于模糊问题的代码
2. dataloader.py
之后所有模型训练使用到的数据都在这个里面，这个文件代码比较多，较为复杂
`    mytestdataset = TamperDataset(using_data={'my_sp': False,
                                              'my_cm': False,
                                              'template_casia_casia': False,
                                              'template_coco_casia': False,
                                              'cod10k': False,
                                              'casia': False,
                                              'coverage': True,
                                              'columb': False,
                                              'negative_coco': False,
                                              'negative_casia': False,
                                              'texture_sp': False,
                                              'texture_cm': False,
                                              }, train_val_test_mode='test',stage_type='stage1')
`
这是我需要使用到的几种类型的数据，需要使用的时候改为True就行，这部分的代码逻辑是:首先使用Mixdata 处理好要使用的数据路径，使用TamperDataset按照pytorch需要的方式处理好数据。
在阅读这部分代码的时候，可以直接运行或者单步调试，但是记得先修改数据集的路径


3. gen_8_map.py
我没有保存8张图，而是动态的生成他们

4. dataloader_blur.py
边缘单独模糊的dataloader,我没有保存模糊后的数据，也是动态的生成，由于pytorch的数据加载器可以提前读取，所以不会影响训练速度

### model
所有关于模型构建的代码都在这里
1.model.py
这是我2020年7月份时候写的代码，照着彪哥的tensorflow模型纯手工写出来的，现在已经没有使用

2. model_two_stage.py

这是我2020年7月份时候写的代码，照着彪哥的tensorflow模型纯手工写出来的，两阶段模型，现在已经没有使用


3. unet_model.py & unet_parts.py
unet的网络代码, parts相当于车轮子，unet_model.py相当于造车，先造车轮子再造车，你去百度搜一下unet的网络结构图可以帮助你理解这一部分的代码
4. unet_*****.py
其他这种形式的代码都是3的变种，通过名字和代码中的注释你可以轻而易举的理解它们，我现在使用的模型是_0306的后缀,0306_3是加了aspp模块的。

### runs
这是保存tensorboard日志的地方

### save_model
这是保存模型参数的地方

### test_tools
所有与测试有关的代码都在这里
1. test.py 
这是我最开始写的test的代码，你需要先阅读这部分代码，这样可以帮助你理解基本的流程
2. stage1_std_test.py
这是第一阶段的测试代码，着无非就是对1进行了二次封装，方便使用而已，但是里面设计到类的继承的一些关系，所有理解起来可能有些费尽，你可以先使用1
3. stage1&2_std_test.py
这是两阶段的测试代码

4. analysis_band_pred.py
这是对第一阶段条带结果的分析代码，现在基本没有使用

5. test_all_two_stage.py
弃用

6. test_one_public_dataset.py
在公开数据集上进行测试的代码

**对于这部分内容，我推荐你先看懂1 再看懂2，然后其他的基本就懂了**

### train
这是所有训练的代码
1. stage1_std_unet.py
训练第一阶段，代码结构由四个函数构成:main;train;val;test
其中main是入口函数，你应该从main开始看;train是训练代码，val是在合成数据测试集上的测试代码，test是在公开数据集上的测试代码，
在test中我只使用了coverage数据集用于跟踪实验过程，并可以用tensorboard实时看到结果，关于tensorboard的使用我也建议你去好好学一下
，这能非常提高效率

2. stage2_std_unet_0306.py
训练第二阶段，这与1意义，只不过是使用了两阶段的网络

3. 其他的代码与1、2 大同小异，你只需要搞懂1、2就行，其中的factor是unet中的一个参数，决定网络的通道数，假设一个特征层的通道在factor=1的时候为10
那么它在factor=2的时候就是20

### utils
这也是一些工具，与代码中并没有关联，代码很短，功能如其名

## functions.py
这就是所有损失函数 和 指标计算的地方,在训练的时候不要进行指标计算，这会很浪费时间

## utils.py
一些小工具，可以不用看，在代码中报错的时候再看
