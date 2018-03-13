#Caffe使用简介

##目录
[TOC]

##下载源码
<http://caffe.berkeleyvision.org/installation.html>
```
git clone https://github.com/BVLC/caffe.git
cd caffe-master
```

##CMake 编译
```
mkdir build
cd build
cmake ..
make all
make install
make
```

##使用caffe库训练LeNet在MNIST
###准确数据集
根据shell文件（wget或gunzip）安装mnist_train_lmdb和mnist_test_lmbd数据文件
```
./data/mnist/get_mnist.sh
./example/mnist/create_mnist.sh
```

LeNet包含CNN，由一个卷积层，一个汇集层，另一个卷积层，然后是一个汇集层，然后是两个完全连接的层
```
/examples/mnist/lenet_train_test.prototxt
```

###定义MNIST网络

mnist 是一个手写数字库，由 DL 大牛YanLeCun进行维护。mnist最初用于支票上的手写数字识别,现在成了DL的入门练习库。针对mnist识别的专门模型是Lenet，算是最早的 CNN模型了。

<http://caffe.berkeleyvision.org/gathered/examples/mnist.html>
**设定网络名称**

```
name: "LeNet"
```

从LMDB文件**读取MNIST的数据**，下面是定义的一个数据层；
name: 表示该层的名称，可随意取
type: 层类型，如果是Data，表示数据来源于LevelDB或LMDB。根据数据的来源不同，数据层的类型也不同。一般在练习的时候，我们都是采用LevelDB或LMDB数据，因此层类型设置为Data。 

Transformations: 数据的预处理，可以将数据变换到定义的范围内。如设置scale为0.00390625，实际上就是1/255, 即将输入数据由0-255归一化到0~1之间 
source：读取lmdb的路径
两个blobs：data blob和label blob

数据层是每个模型的最底层，是模型的入口，仅提供数据的输入，也提供数据从Blobs转换成别的格式进行保存输出。通常数据的预处理(如减去均值,缩放,裁剪和镜像等)，也在这层设置参数实现。 

数据来源可以来自高效的数据库(如LevelDB和LMDB)，也可以直接来 于内存。如果 是很注重效率的话，数 据也可来 磁盘的hdf5 件和图 格式 件。 
```
layer {
  name: "mnist"	
  type: "Data"
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "mnist_train_lmdb"
    backend: LMDB
    batch_size: 64
  }
  top: "data"
  top: "label"
}
```

**内存数据(MemoryData)**

batch_size: 每一次处理的数据个数，比如2 
channels: 通道数 
height: 高度 
width: 宽度 

```
layer {
  top: "data"
  top: "label"
  name: "memory_data"
  type: "MemoryData"
  memory_data_param{
    batch_size: 2
    height: 100
    width: 100
    channels: 1
  }
  transform_param {
    scale: 0.0078125
    mean_file: "mean.proto"
    mirror: false
} } 
```

**HDF5数据（HDF5Data）**
```
layer {
  name: "data"
  type: "HDF5Data"
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "examples/hdf5_classification/data/train.txt"
    batch_size: 10
  }
} 
```


**定义第一个卷积层**
设定20个通道的输出，卷积核大小是5，步长是1
The fillers allow us to randomly initialize the value of the weights and bias. For the weight filler, we will use the xavier algorithm that automatically determines the scale of initialization based on the number of input and output neurons. For the bias filler, we will simply initialize it as constant, with the default filling value 0.

lr_mults are the learning rate adjustments for the layer’s learnable parameters. In this case, we will set the weight learning rate to be the same as the learning rate given by the solver during runtime, and the bias learning rate to be twice as large as that - this usually leads to better convergence rates

```
layer {
  name: "conv1"
  type: "Convolution"
  param { lr_mult: 1 }
  param { lr_mult: 2 }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
  bottom: "data"
  top: "conv1"
}
```

###定义MNIST Solver

test_iter: 测试样本，训练的图片个数/Batch_size
test_interval: 迭代次数
base_lr：lr是learning rate，学习速率，数据量较少的时候需要设置小一点，防止过早收敛
max_iter： 最大迭代

```
# The train/test net protocol buffer definition
net: "examples/mnist/lenet_train_test.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100
# Carry out testing every 500 training iterations.
test_interval: 500
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005
# The learning rate policy
lr_policy: "inv"
gamma: 0.0001
power: 0.75
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 10000
# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"
# solver mode: CPU or GPU
solver_mode: GPU
```

###训练自己的模型

**归类**好需要的数据集，放置在统一的文件夹目录
生成对应的trail.txt和test.txt文件，用于转图片为lmdb文件
```
import os

root = "/Users/qiyun/Downloads/caffe-1.0/data/test_mnist/"
data = 'train'
print(root)
path = os.listdir(root + data)
path.sort()
file = open(root + 'train.txt', 'w')

i = 0
j = 0

for line in path:
    str = root + data + '/' + line
    print(str)

    if 'DS_Store' in str:
        print("error file path ....")
    else:
        for child in os.listdir(str):
            str1 = data + '/' + line + '/' + child
            filePath = root + str1
            fix = os.path.splitext(str1)

            print("####### in  " + filePath)
            print(fix)
            d = '%s' %(i)
            e = '%s' %(j)
            t = fix[0] + fix[1] + ' ' + e
            #t = data + '/' + line + '/' + "img" + d + fix[1] + ' ' + e
            print(t)
            file.write(t +'\n')
            outpath = root + t;
            print("####### out  " + outpath)
            #os.rename(filePath, newfile)
            i = i + 1
        j = j + 1

file.close()

```

**批量处理图片**
LeNet训练集的图片需要28x28的尺寸，各个训练网络都有特定尺寸要求
如果图片尺寸不合适，可以使用下面shell脚步进行处理（需要安装ImageMagick）

```
#!/usr/bin/env bash

src_dir=/Users/qiyun/Downloads/caffe-1.0/data/my_mnist/train/
number=1
dir=`ls -1 $src_dir`
for dir_name in `ls -1 $src_dir`;
do
    if [ -d $src_dir$dir_name ]
    then
        echo $src_dir$dir_name
        number=1
        for file_name in `ls -l $src_dir$dir_name | grep ^- | awk '{print $9}'`;
        do
            echo $src_dir$dir_name"/"$file_name
            convert +profile '*' $src_dir$dir_name"/"$file_name -quality 100 -resize '28x28!' -gravity Center -crop 28x28+0+0 +repage $src_dir$dir_name"/"$number"_img"".jpg"
            #mv $src_dir$dir_name"/"$file_name $src_dir$dir_name"/"$file_name
            rm -rf $src_dir$dir_name"/"$file_name
            echo $number
            let number=number+1
        done
    fi
done

```

###生成LMDB文件

**需要修改对应的路径**

```
#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=data/my_mnist
DATA=/Users/qiyun/Downloads/caffe-1.0/data/my_mnist/
TOOLS=build/tools

TRAIN_DATA_ROOT=/Users/qiyun/Downloads/caffe-1.0/data/my_mnist/
VAL_DATA_ROOT=/Users/qiyun/Downloads/caffe-1.0/data/my_mnist/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=28
  RESIZE_WIDTH=28
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/val_lmdb

echo "Done."
```

**生成均值文件，用于最后的预测数据**

Caffe 中使用的均值数据格式是 binaryproto, 作者为我们提供了一个计算均值的文件 compute_image_mean.cpp，放在 Caffe 根目录下的 tools 文件夹里面。编译后的可执行体放在 build/tools/ 下面，我们直接调用就可以了。

```
./build/tools/compute_image_mean examples/mnist/mnist_train_lmdb examples/mnist/mean.binaryproto
```

```
#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=data/my_mnist
DATA=/Users/qiyun/Downloads/caffe-1.0/data/my_mnist
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
```

**开始训练**

```
#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=data/my_mnist/lenet_solver.prototxt
```

##使用

**测试**

训练完成后，可以直接使用build下的代码进行测试，命令如下

```
#!/usr/bin/env sh
set -e

./build/examples/cpp_classification/classification.bin data/my_mnist/lenet.prototxt ./data/my_mnist/pubg_classify.caffemodel data/my_mnist/pubg_imagenet_mean.binaryproto data/my_mnist/pubg_labels.txt /Users/qiyun/Desktop/outputImage.jpg
```

**结果**

```
yuns-iMac:caffe-1.0 qiyun$ ./data/my_mnist/TestImage.sh 
---------- Prediction for /Users/qiyun/Desktop/outputImage.jpg ----------
1.0000 - "1_(生存)"
0.0000 - "2_(生存 english)"
0.0000 - "0_(存活)"
0.0000 - "4_(不在游戏中)"
0.0000 - "3_(加入)"
```

##附录
Google Protocol Buffer的使用和原理
https://www.ibm.com/developerworks/cn/linux/l-cn-gpb/index.html
makefile编译调用Caffe框架的C++程序
http://yongyuan.name/blog/compiling-cpp-code-using-caffe.html
https://github.com/Tencent/ncnn
ncnn 组件使用指北 alexnet
https://github.com/Tencent/ncnn/wiki/ncnn-组件使用指北-alexnet
python使用caffe
http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/

