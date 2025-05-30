#!/bin/bash

# 创建 output 文件夹结构
mkdir -p output/test output/fa1

# 映射编号到对应命令和输出路径
for ARG in "$@"
do
    case $ARG in
        0)
            python3 gpt149.py test > output/test/0-1.txt
            ;;
        1)
            python3 gpt149.py fa1 > output/fa1/1-1.txt
            python3 gpt149.py fa1 -N 512 > output/fa1/1-2.txt
            python3 gpt149.py fa1 -N 2048 > output/fa1/1-3.txt
            ;;
        *)
            echo "Unknown part number: $ARG"
            echo "Usage: $0 [0, 1] (0=test, 1=fa1)"
            exit 1
            ;;
    esac
done