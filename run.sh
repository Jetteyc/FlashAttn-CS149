#!/bin/bash

# 创建 output 文件夹结构
mkdir -p output/warmup output/part1 output/part2 output/part3 output/part4

# 映射编号到对应命令和输出路径
for ARG in "$@"
do
    case $ARG in
        0)
            python3 gpt149.py 4Daccess > output/warmup/0-1.txt
            python3 gpt149.py 4Daccess > output/warmup/0-2.txt
            python3 gpt149.py 4Daccess > output/warmup/0-3.txt
            ;;
        1)
            python3 gpt149.py part1 > output/part1/1-1.txt
            python3 gpt149.py part1 -N 512 > output/part1/1-2.txt
            python3 gpt149.py part1 -N 2048 > output/part1/1-3.txt
            ;;
        2)
            python3 gpt149.py part2 > output/part2/2-1.txt
            python3 gpt149.py part2 -N 512 > output/part2/2-2t.txt
            python3 gpt149.py part2 -N 2048 > output/part2/2-3t.txt
            ;;
        3)
            python3 gpt149.py part3 > output/part3/3-1.txt
            python3 gpt149.py part3 -N 512 > output/part3/3-2t.txt
            python3 gpt149.py part3 -N 2048 > output/part3/3-3t.txt
            ;;
        4)
            python3 gpt149.py part4 > output/part4/4-1.txt
            python3 gpt149.py part4 -N 512 > output/part4/4-2.txt
            python3 gpt149.py part4 -N 2048 > output/part4/4-3.txt
            python3 gpt149.py part4 -N 2048 -br 128 -bc 128 > output/part4/4-4.txt
            python3 gpt149.py part4 -N 2048 -br 50 -bc 50 > output/part4/4-5.txt
            ;;
        *)
            echo "Unknown part number: $ARG"
            echo "Usage: $0 [0 1 2 3 4 5] (0=warmup, 1=part1, ..., 5=vectorization)"
            exit 1
            ;;
    esac
done