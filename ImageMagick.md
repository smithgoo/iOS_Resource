#ImageMagick 7.0.7-16使用

**相关链接**
* HomePage: <https://www.imagemagick.org/script/index.php>
* Download: <https://github.com/ImageMagick/ImageMagick>

##安装
	•brew install ImageMagick
##使用

**Edge Detection**
```
convert input.png -colorspace gray -canny 0x1+10%+30% output.png
```

**Compositon Image**

把一张图片按80的质量去压缩(jpg的压缩参数),同时按图片比例非强制缩放成不超过280x140的图片.居中裁剪280x140,去掉图片裁减后的空白和图片exif信息,通常这种指令是为了保证图片大小正好为280x140
```
convert +profile '*' /Users/qiyun/Desktop/屏幕快照\ 2018-01-03\ 下午2.04.04.png -quality 80 -resize '280x140^>' -gravity Center -crop 280x140+0+0 +repage /Users/qiyun/Desktop/新图片.jpg
```
`
-quality   图片质量,jpg默认99,png默认75
-resize
100x100      高度和宽度比例保留最高值，高比不变
100x100^     高度和宽度比例保留最低值，宽高比不变
100x100!     宽度和高度强制转换，忽视宽高比
100x100>     更改长宽，当图片长或宽超过规定的尺寸
100x100<     更改长宽 只有当图片长宽都超过规定的尺寸
100x100^>    更改长宽，当图片长或宽超过规定的尺寸。高度和宽度比例保留最低值
100x100^<    更改长宽，只有当图片长宽都超过规定的尺寸。高度和宽度比例保留最低值
100          按指定的宽度缩放，保持宽高比例
x100         按指定高度缩放，保持宽高比
-gravity NorthWest, North, NorthEast, West, Center, East,  SouthWest, South, SouthEast截取用的定位指令,定位截取区域在图片中的方位
-crop 200x200+0+0 截取用的截取指令 ,在用定位指令后,按后两位的偏移值偏移截取范围左上角的像素后,再按前两位的数值,从左上角开始截取相应大小的图片
+repage         去掉图片裁减后的空白
-dissolve 30    设定组合图片透明度dissolve示例
+/-profile *    去掉/添加图片exif信息
`

**把原始图片分割成多张小图**

	•convert src.jpg -crop 100x100 dest.jpg

假设src.jpg的大小是300x200,执行命令后将得到名为dest-0.jpg、dest-1.jpg...dest-5.jpg的6张大小为100x100的小图片。注意如果尺寸不是目标图片的整数倍，那么右边缘和下边缘的一部分图片就用实际尺寸

**在原始图片上剪裁一张指定尺寸的小图**

	•convert src.jpg -crop 100x80+50+30 dest.jpg
在原始图片的上距离上部30像素左部50为起点的位置,分别向右向下截取一块大小为100x80的图片。如果x相对于坐标，宽度不够100，那就取实际值。

	•convert src.jpg -gravity center -crop 100x80+0+0 dest.jpg
在原始图上截取中心部分一块100x80的图片

	•convert src.jpg -gravity southeast -crop 100x80+10+5 dest.jpg
在原始图上截取右下角距离下边缘10个像素，右边缘5个像素一块100x80的图片

**图片进行反色处理**

	•convert -negate src.jpg negate.jpg

##示例

**转换rgb为灰度图**

```
计算公式 >> Grey = (R*38 + G*75 + B*15)>> 7
convert /Users/qiyun/Desktop/屏幕快照\ 2018-02-23\ 下午2.51.29.png -set colorspace Gray -separate -average 1111.jpeg
```

**-crop参数是从一个图片截取一个指定区域的子图片**
 ```
格式如下:convert -crop widthxheight{+-}x{+-}y{%}
```
 *width 子图片宽度* *height 子图片高度*
 x 为正数时为从区域左上角的x坐标,为负数时,左上角坐标为0,然后从截出的子图片右边减去x象素宽度. y 为正数时为从区域左上角的y坐标,为负数时,左上角坐标为0,然后从截出的子图片上边减去y象素高度.  如convert -crop 300x400+10+10 src.jpg dest.jpg 从src.jpg坐标为x:10 y:10截取300x400的图片存为dest.jpg convert -crop 300x400-10+10 src.jpg dest.jpg 从src.jpg坐标为x:0 y:10截取290x400的图片存为dest.jpg

##总结


