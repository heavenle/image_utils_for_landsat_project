# image_utils_for_landsat_project
针对于landsat项目，所编写的图像数据处理代码。

landsat 是一个遥感数据，其分辨率较高，无法直接进行网络训练。所以本项目主要是针对遥感数据进行一系列预处理来适配网络输入。目前以Unet2系列为标准。

文件内容：
1. crop.py  裁剪文件，将高分辨率大图，以固定尺寸的滑框进行裁剪。
2. Filter.py 将数据中少数类样本以及全黑图像进行筛选。
3. Norm.py 归一化文件，分别对图像中每个通道进行归一化。
4. ShowMaskNpy.py 将每个标签中对应的类别以RGB颜色进行显示。
5. Statistical.py 统计所有标签中没个类别的占比，并绘制成饼状图。
6. ResizeImg.py 将每种resize方式都显示出来，判断那种结果适合遥感影像的缩放。
7. DataRandomCat.py 通过混拼的方式进行数据增强。即将图像等分成不同大小的图。然后将所有小图进行随机拼接。
8. ExtractForeGround.py 通过提取少数类别的像素前景来进行数据增强，通常配合混拼使用。
9. pspnet.py 各种数据增强方法。例如随机旋转，缩放，裁剪等。

