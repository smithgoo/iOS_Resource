##目录
[TOC]

#Tensorflow V1.2

GitHub地址: <https://github.com/tensorflow/tensorflow>

```
TensorFlow is an open source software library for numerical computation using data flow graphs. 
The graph nodes represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) that flow between them. 
This flexible architecture lets you deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device without rewriting code. 
TensorFlow also includes TensorBoard, a data visualization toolkit.
```


##环境设置
Linux CPU-only: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp27-none-linux_x86_64.whl) [(build history)](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/) / [Python 3.4](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp34-cp34m-linux_x86_64.whl) [(build history)](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/) / [Python 3.5](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp35-cp35m-linux_x86_64.whl) [(build history)](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/) / [Python 3.6](http://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.6,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp36-cp36m-linux_x86_64.whl) [(build history)](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.6,label=cpu-slave/)

##安装编译

TensorFlow Python API 依赖 Python 2.7 版本.
在 Linux 和 Mac 下最简单的安装方式, 是使用 [pip](https://pypi.python.org/pypi/pip) 安装.
如果在安装过程中遇到错误, 请查阅 [常见问题](http://www.tensorfly.cn/tfdoc/get_started/os_setup.html#common_install_problems). 为了简化安装步骤, 建议使用 virtualenv, 教程见 [这里](http://www.tensorfly.cn/tfdoc/get_started/os_setup.html#virtualenv_install).

####Ubuntu/Linux
```
# 仅使用 CPU 的版本
$ pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl

# 开启 GPU 支持的版本 (安装该版本的前提是已经安装了 CUDA sdk)
$ pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.5.0-cp27-none-linux_x86_64.whl
```

####Mac OS X

在 OS X 系统上, 推荐先安装 [homebrew](http://brew.sh/), 然后执行 brew install python, 以便能够使用 homebrew 中的 Python 安装 TensorFlow. 另外一种推荐的方式是在 [virtualenv](http://www.tensorfly.cn/tfdoc/get_started/os_setup.html#virtualenv_install) 中安装 TensorFlow.

[安装步骤](https://docs.bazel.build/versions/master/install-os-x.html)
```
# 当前版本只支持 CPU
$ pip install https://storage.googleapis.com/tensorflow/mac/tensorflow-0.5.0-py2-none-any.whl
```

####基于 Docker 的安装

也支持通过 [Docker](http://docker.com/) 运行 TensorFlow. 该方式的优点是不用操心软件依赖问题.
首先, [安装 Docker](http://docs.docker.com/engine/installation/). 一旦 Docker 已经启动运行, 可以通过命令启动一个容器:

```
$ docker run -it b.gcr.io/tensorflow/tensorflow
```

####基于 VirtualEnv 的安装

推荐使用 [virtualenv](https://pypi.python.org/pypi/virtualenv) 创建一个隔离的容器, 来安装 TensorFlow. 这是可选的, 但是这样做能使排查安装问题变得更容易.
首先, 安装所有必备工具:

```
# 在 Linux 上:
$ sudo apt-get install python-pip python-dev python-virtualenv

# 在 Mac 上:
$ sudo easy_install pip  # 如果还没有安装 pip
$ sudo pip install --upgrade virtualenv
```

接下来, 建立一个全新的 virtualenv 环境. 为了将环境建在 ~/tensorflow 目录下, 执行:

```
$ virtualenv --system-site-packages ~/tensorflow
$ cd ~/tensorflow
```

然后, 激活 virtualenv:

```
$ source bin/activate  # 如果使用 bash
$ source bin/activate.csh  # 如果使用 csh
(tensorflow)$  # 终端提示符应该发生变化
```

在 virtualenv 内, 安装 TensorFlow:

```
(tensorflow)$ pip install --upgrade <$url_to_binary.whl>
```

接下来, 使用类似命令运行 TensorFlow 程序:

```
(tensorflow)$ cd tensorflow/models/image/mnist
(tensorflow)$ python convolutional.py

# 当使用完 TensorFlow
(tensorflow)$ deactivate  # 停用 virtualenv

$  # 你的命令提示符会恢复原样
```

####从源码安装

克隆 TensorFlow 仓库

```
$ git clone --recurse-submodules https://github.com/tensorflow/tensorflow
```
--recurse-submodules 参数是必须得, 用于获取 TesorFlow 依赖的 protobuf 库.

####Linux 安装

**安装 Bazel**
首先依照 [教程](http://bazel.io/docs/install.html) 安装 Bazel 的依赖. 然后使用下列命令下载和编译 Bazel 的源码:

```
$ git clone https://github.com/bazelbuild/bazel.git
$ cd bazel
$ git checkout tags/0.1.0
$ ./compile.sh
```

上面命令中拉取的代码标签为 0.1.0, 兼容 Tensorflow 目前版本. bazel 的HEAD 版本 (即最新版本) 在这里可能不稳定.
将执行路径 output/bazel 添加到 $PATH 环境变量中.

安装其他依赖

```
$ sudo apt-get install python-numpy swig python-dev
```

**可选: 安装 CUDA (在 Linux 上开启 GPU 支持)**
为了编译并运行能够使用 GPU 的 TensorFlow, 需要先安装 NVIDIA 提供的 Cuda Toolkit 7.0 和 CUDNN 6.5 V2.
TensorFlow 的 GPU 特性只支持 NVidia Compute Capability >= 3.5 的显卡. 被支持的显卡 包括但不限于:

	•	NVidia Titan
	•	NVidia Titan X
	•	NVidia K20
	•	NVidia K40
	
**下载并安装 Cuda Toolkit 7.0**
[下载地址](https://developer.nvidia.com/cuda-toolkit-70)

**下载并安装 CUDNN Toolkit 6.5**
[下载地址](https://developer.nvidia.com/rdp/cudnn-archive)

解压并拷贝 CUDNN 文件到 Cuda Toolkit 7.0 安装路径下. 假设 Cuda Toolkit 7.0 安装 在 /usr/local/cuda, 执行以下命令:

```
tar xvzf cudnn-6.5-linux-x64-v2.tgz
sudo cp cudnn-6.5-linux-x64-v2/cudnn.h /usr/local/cuda/include
sudo cp cudnn-6.5-linux-x64-v2/libcudnn* /usr/local/cuda/lib64
```

**配置 TensorFlow 的 Cuba 选项**
从源码树的根路径执行:

```
$ ./configure
Do you wish to bulid TensorFlow with GPU support? [y/n] y
GPU support will be enabled for TensorFlow

Please specify the location where CUDA 7.0 toolkit is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda

Please specify the location where CUDNN 6.5 V2 library is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda

Setting up Cuda include
Setting up Cuda lib64
Setting up Cuda bin
Setting up Cuda nvvm
Configuration finished
```

这些配置将建立到系统 Cuda 库的符号链接. 每当 Cuda 库的路径发生变更时, 必须重新执行上述 步骤, 否则无法调用 bazel 编译命令.

**编译目标程序, 开启 GPU 支持**
从源码树的根路径执行

```
$ bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer

$ bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu
# 大量的输出信息. 这个例子用 GPU 迭代计算一个 2x2 矩阵的主特征值 (major eigenvalue).
# 最后几行输出和下面的信息类似.
000009/000005 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000006/000001 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
000009/000009 lambda = 2.000000 x = [0.894427 -0.447214] y = [1.788854 -0.894427]
```

注意, GPU 支持需通过编译选项 "--config=cuda" 开启.

**已知问题**

```
尽管可以在同一个源码树下编译开启 Cuda 支持和禁用 Cuda 支持的版本, 我们还是推荐在 在切换这两种不同的编译配置时, 使用 "bazel clean" 清理环境.

在执行 bazel 编译前必须先运行 configure, 否则编译会失败并提示错误信息. 未来, 我们可能考虑将 configure 步骤包含在编译过程中, 以简化整个过程, 前提是 bazel 能够提供新的特性支持这样.
```

##Example

####第一个 TensorFlow 程序

(可选) 启用 GPU 支持

如果使用 pip 二进制包安装了开启 GPU 支持的 TensorFlow, 必须确保 系统里安装了正确的 CUDA sdk 和 CUDNN 版本. 请参间 [CUDA 安装教程](http://www.tensorfly.cn/tfdoc/get_started/os_setup.html#install_cuda)
还需要设置 LD_LIBRARY_PATH 和 CUDA_HOME 环境变量. 可以考虑将下面的命令 添加到 ~/.bash_profile 文件中, 这样每次登陆后自动生效. 
注意, 下面的命令 假定 CUDA 安装目录为 /usr/local/cuda:

```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
```

####运行 TensorFlow

打开一个 python 终端:

```
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
```

####安装Anaconda

对于Mac、Linux系统，Anaconda安装好后，实际上就是在主目录下多了个文件夹（~/anaconda）而已，Windows会写入注册表。安装时，安装程序会把bin目录加入PATH（Linux/Mac写入~/.bashrc，Windows添加到系统变量PATH），这些操作也完全可以自己完成。以Linux/Mac为例，安装完成后设置PATH的操作是
```
# 将anaconda的bin目录加入PATH，根据版本不同，也可能是~/anaconda3/bin
echo 'export PATH="~/anaconda2/bin:$PATH"' >> ~/.bashrc
# 更新bashrc以立即生效
source ~/.bashrc

```


切换python版本
```
# 创建一个名为python34的环境，指定Python版本是3.4（不用管是3.4.x，conda会为我们自动寻找3.4.x中的最新版本）
conda create --name python34 python=3.4

# 安装好后，使用activate激活某个环境
activate python34 # for Windows
source activate python34 # for Linux & Mac
# 激活后，会发现terminal输入的地方多了python34的字样，实际上，此时系统做的事情就是把默认2.7环境从PATH中去除，再把3.4对应的命令加入PATH

# 此时，再次输入
python --version
# 可以得到`Python 3.4.5 :: Anaconda 4.1.1 (64-bit)`，即系统已经切换到了3.4的环境

# 如果想返回默认的python 2.7环境，运行
deactivate python34 # for Windows
source deactivate python34 # for Linux & Mac

# 删除一个已有的环境
conda remove --name python34 --all
```

##Inception

[Inception (GoogLeNet)](https://arxiv.org/pdf/1512.00567v3.pdf)是Google 2014年发布的Deep Convolutional Neural Network，其它几个流行的CNN网络还有[QuocNet](http://static.googleusercontent.com/media/research.google.com/en//archive/unsupervised_icml2012.pdf)、[AlexNet](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)、[BN-Inception-v2](http://arxiv.org/abs/1502.03167)、[VGG](https://arxiv.org/abs/1409.1556)、[ResNet](https://arxiv.org/pdf/1512.03385v1.pdf)等等。

Inception V3模型源码定义：[tensorflow/contrib/slim/python/slim/nets/inception_v3.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py)

##总结

###相关链接
[Tensorflow 中文社区](http://www.tensorfly.cn/tfdoc/get_started/introduction.html)
[Anaconda 下载](https://www.anaconda.com/download/#macos)
[Anaconda 使用总结](https://www.jianshu.com/p/2f3be7781451)
[TensorFlow 安装教程](http://blog.csdn.net/lijjianqing/article/details/54671503)
[Tensorflow 安装 和集成到IDEA](http://blog.csdn.net/lijjianqing/article/details/54671503)
[Tensorflow 使用实战](http://blog.topspeedsnail.com)

