# cuda
### GPU编程简介
CUDA开发主要是是以主机（CPU）为出发点的应用程序可以调用CUDA运行时API、CUDA驱动API及一些已有的CUDA库。所有这些调用都将利用设备（GPU）的硬件资源。

![image-20250412094034939](C:\Users\Lenovo1\AppData\Roaming\Typora\typora-user-images\image-20250412094034939.png)

### 开发环境搭建

win11环境下

VS2022

CUDA（需要cuda studio integration）

CUDNN（深度学习加速）

测试是否安装成功

```shell
nvcc -V
```

查看显卡信息

```shell
nvidia-smi
```

### CUDA开发流程

```c++
int main(viod)
{
    主机代码
    核函数调用
    主机代码
    returen
}
```

CUDA中的核函数与C＋＋中的函数是类似的，但＿个显著的差别是它必须 被限定词global修饰。其中global前后是双下划线。另外核 函数的返回类型必须是空类型。限定符global和void的次序可随意。

```C++
__global__ void hello() {
	printf("Hello World from GPU!\n");
}
```

调用核函数

```c++
#helloworld.cu
#include<stdio.h>

void __global__  hello() {
	printf("Hello World from GPU!\n");
}
int main() {
	// Launch kernel on GPU
	hello << <1, 5 >> > ();
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	return 0;
}
```

主机在调用一个核函数时必须指明需要在设备中指派多少个线程，否则设备不知道如何工作。三括号中的数就是用来指明核函数中的线程数目及排列情况的。<<<grid_size,block_size>>>

#### 网格，线程块，线程

![image-20250412110415035](C:\Users\Lenovo1\AppData\Roaming\Typora\typora-user-images\image-20250412110415035.png)

由一个内核启动所产生的所有线程统称一个**网格**（Grid），**同一网格中的所有线程共享相同的全局内存空间**。

一个网格由多个**线程块**（Block）构成。一个线程块由**一组线程**（Thread）构成。线程网格和线程块从逻辑上代表了一个核函数的线程层次结构，这种组织方式可以帮助我们有效地利用资源，优化性能。CUDA编程中，我们可以组织三维的线程网格和线程块，具体如何组织，一般是和我们需要处理的数据有关。上面这个示意图展示的是一个包含二维线程块的二维线程网格。



#### 线程索

可以为一个核函数指派多个线程，而这些线程的组织结构由执行配置决定

<<<grid_size,block_size>>>

这里的grid_size（网格大小）和block_size（线程块大小）一般来说是一个结构体类型的变量，但也可以是—个普通的整型变量，这两个整型变量的乘积就是被调用核函数中总的线程数。

一般来说只要线程数比GPU中的计算核心数（几百至几千 个）多几倍时就有可能充分地利用GPU中的全部计算资源。

每个线程在核函数中都有一个唯一的身份标识。由于我们用两个参数指定了线程数目，每个线程的身份可由两个参数确定，在核函数内部，程序是知道执行配置参数grid_size和block_size的值的，这两个值分别保存于gridDim.x和blockDim.x中，如果网格和线程块是二维或三维，则还有.y和.z参数

