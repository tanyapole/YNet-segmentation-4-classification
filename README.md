# YNet-segmentation-4-classification
Using segmentation masks to improve classification with a YNet architercture

example of usage
python my_main.py --N 5  --lr 0.0001 --batch_size 30 --attribute attribute_globules attribute_milia_like_cyst  --workers 1 --image_path /data/ISIC/ --normalize --model_path /data/ISIC/model_ynet_pretrain_2cl/ --pretrained
