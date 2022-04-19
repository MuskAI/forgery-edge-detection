## 数据处理

对于第一阶段的数据处理我决定新增一些数据增广方法，但是在篡改检测任务中，这些方法应该被慎重考虑

要考虑限度的问题，因此，选择下面这些方法：


1. 随机crop
2. RandomHorizontalFlip
3. RandomVerticalFlip
4. 加噪声（小范围
5. 加模糊（小范围
6. RandomCrop