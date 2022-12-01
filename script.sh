$ python train.py --img_backbone s3d --shift --shift_place block --batch_size 8 --lr 1e-4 --lr_sched True --load_weight enet_transformer.pt

$ CUDA_VISIBLE_DEVICES=0 python train.py --img_backbone combo_s3d --shift --lr_sched True --clip_size 128 --batch_size 1