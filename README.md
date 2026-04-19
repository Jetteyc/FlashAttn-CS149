# 此分支用于完成作业-ispc优化的FAv1

## 项目结构
仓库根目录的 model.py、sample.py 基本跟 nanoGPT 一套，只是把 attention 改成会走module.cpp，可选ispc 编出来的 module_ispc.o，和PyTorch 参考结果对齐检查（注意这里没实现1/sqrt(d) 缩放）

gpt149.py 是单独的 micro-benchmark，会 load 同一份 module.cpp 和参考实现 module_ref 比对错和耗时。run.sh、run_ispc.sh 用来批量跑实验，日志落在 output/、output_ispc/ 下面。

## 环境
参考https://github.com/stanford-cs149/cs149gpt/issues/2
```py
conda create -n gpt149
conda activate gpt149
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 cpuonly python=3.10 numpy=1.26 ninja tiktoken -c pytorch -c conda-forge
python3 -m pip install --upgrade --force-reinstall "setuptools<70"
```

## 2D Accessor与4D Accessor
在module.cpp补充fourDimRead与fourDimWrite。
测试：
```py
python3 gpt149.py 4Daccess
```

## Part 1:简单attention
在module.cpp补充myNaiveAttention。
数据是(b,h,n,d)格式但是中间的QK_t和softmax(QK^t)是(n,d)
```py
python3 gpt149.py part1
```
ref:
Pytorch Execution Time: 1.123816728591919 
Manual Execution Time:  0.09308695793151855
cpu time:  93.077ms
mem usage:  4718592 bytes

my:
Pytorch Execution Time: 1.1280500888824463 
Manual Execution Time:  0.08842945098876953 
cpu time:  88.418ms
mem usage:  4718592 bytes

## Part 2:仅分块
在module.cpp补充myUnfusedAttentionBlocked。
仅对矩阵乘法分块处理，softmax不变。

Q、K、QK_t各`L*L*4`字节，共12L^2,
`lscpu`看cpu数据，这里只做一层分块，让其尽量在l1 cache里，也就是L1d，令`12L^2<L1d`,考虑一些其他的开销L可能需要进一步缩小。如果多层的分块可以考虑里用l2和l3。
经过测试L=32效率最高。

每个float 4byte，每个cache line 64bytes，因此每个dram读取搬运16个float。


```cpp
for i in 0 to N-1:       // 遍历 Q 的每一行
    for j in 0 to N-1:   // 遍历 K 的每一行
        sum = dot_product(Q[i], K[j])
        O[i,j] = sum
```
原版中访问Q需要`N*d/16`个cache line，访问一次完整的K也要`N*d/16`个cache line，但是i每次切换K都要完整访问一次，就是`N*N*d/16`个cache line，所以总访问量`O(N*N*d/16)`。
分块后，每个块都是`L*L/16`，每个Q块只访问了一次，对Q的总访问是`(N/L)*(d/L)*L*L/16=N*d/16`,但是K块每个都访问了N/L次，所以总共就是`N*N*d/16L`，总体比例是1/L
```py
python3 gpt149.py part2
```
ref:
Pytorch Execution Time: 1.091418743133545 
Manual Execution Time:  0.08716082572937012 
pu time:  87.149ms
mem usage:  4718592 bytes
my:
Pytorch Execution Time: 1.0957069396972656 
Manual Execution Time:  0.0744538307189941
cpu time:  74.438ms
mem usage:  4718592 bytes

## Part 3:融合与OpenMP
先计算QK结果的第一行，直接softmax，然后和V相乘，得到最终结果的第一行，然后再计算下一行。
可见每一行计算完全独立,b、h维度也完全独立，这三层for循环可用OpenMP优化
openmp的使用：
```cpp
#pragma omp parallel for collapse(3)
```
在for循环里面用at::Tensor ORowTensor = temp.index({torch::indexing::Slice(omp_get_thread_num(), torch::indexing::None)});每个任务要独占一个ORow数组，线程数通常是24，可由环境变量OMP_NUM_THREADS指定。
```py
python3 gpt149.py part3
```
ref:
Pytorch Execution Time: 1.0916287899017334 
Manual Execution Time:  0.02878284454345703 
cpu time:  28.773ms
mem usage:  557056 bytes
my:
Pytorch Execution Time: 1.0933377742767334 
Manual Execution Time:  0.034758567810058594
cpu time:  34.749ms
mem usage:  557056 bytes

## Part 4:flash attn
KV在外层做循环，Q在内层。
最终Oi是以逐个br*d来更新的，内层循环完一次就会更新一次整个O，外层循环共更新了Tc次O。
```py
python3 gpt149.py part4
```
ref:
Pytorch Execution Time: 1.143669843673706 
Manual Execution Time:  0.25463151931762695
cpu time:  254.153ms
mem usage:  524288 bytes
my:
Pytorch Execution Time: 1.1569817066192627 
Manual Execution Time:  0.07818055152893066
cpu time:  77.819ms
mem usage:  524288 bytes
part3快于part4的原因，可能是因为part3的多线程会快于part4的单线程，而part4的方式又很难方便地加入openmp，只能通过ispc或者cuda来进行优化。