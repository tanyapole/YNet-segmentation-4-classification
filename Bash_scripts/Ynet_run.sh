#!/bin/bash
python my_main.py --pretrained --N 1 --lr 0.0001 --batch_size 20 --attribute attribute_globules attribute_milia_like_cyst --n_epochs 100 --workers 2 --image_path ./$1 --normalize
