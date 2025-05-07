#!/bin/bash

# 创建 output_ispc 文件夹结构
mkdir -p output_ispc/part1 output_ispc/part2 output_ispc/part3 output_ispc/part4

ispc -O3 --target=avx2-i32x8 --arch=x86-64 --pic module.ispc -h module_ispc.h -o module_ispc.o 

# 映射编号到对应命令和输出路径
for ARG in "$@"
do
    case $ARG in
        0)
            python3 gpt149.py 4Daccess > output_ispc/warmup/0-1.txt
            python3 gpt149.py 4Daccess > output_ispc/warmup/0-2.txt
            python3 gpt149.py 4Daccess > output_ispc/warmup/0-3.txt
            ;;
        1)
            python3 gpt149.py part1 > output_ispc/part1/1-1.txt
            python3 gpt149.py part1 -N 512 > output_ispc/part1/1-2.txt
            python3 gpt149.py part1 -N 2048 > output_ispc/part1/1-3.txt
            ;;
        2)
            python3 gpt149.py part2 > output_ispc/part2/2-1.txt
            python3 gpt149.py part2 -N 512 > output_ispc/part2/2-2t.txt
            python3 gpt149.py part2 -N 2048 > output_ispc/part2/2-3t.txt
            ;;
        3)
            python3 gpt149.py part3 > output_ispc/part3/3-1.txt
            python3 gpt149.py part3 -N 512 > output_ispc/part3/3-2t.txt
            python3 gpt149.py part3 -N 2048 > output_ispc/part3/3-3t.txt
            ;;
        4)
            python3 gpt149.py part4 > output_ispc/part4/4-1.txt
            python3 gpt149.py part4 -N 512 > output_ispc/part4/4-2.txt
            python3 gpt149.py part4 -N 2048 > output_ispc/part4/4-3.txt
            python3 gpt149.py part4 -N 2048 -br 128 -bc 128 > output_ispc/part4/4-4.txt
            python3 gpt149.py part4 -N 2048 -br 50 -bc 50 > output_ispc/part4/4-5.txt
            ;;
        *)
            echo "Unknown part number: $ARG"
            echo "Usage: $0 [0 1 2 3 4 5] (0=warmup, 1=part1, ..., 5=vectorization)"
            exit 1
            ;;
    esac
done