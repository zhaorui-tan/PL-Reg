DATASET="VLCS"
DATADIR="/Data_PHD/phd22_zhaorui_tan/data_raw/dg_data/DG"
# reproduce baselines, including ERM, CE, MIRO, GMDG
#CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.0  --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.0  --d_shift 0.  --mask_range 0.5 --confidence 0.    --lr_mult 2.   --low_degree .0
#CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.0  --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.0  --d_shift 0.  --mask_range 0.5 --confidence 0.1   --lr_mult 2.   --low_degree .0
#CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.01 --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.0  --d_shift 0.  --mask_range 0.5 --confidence 0.    --lr_mult 2.   --low_degree .0
#CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.01 --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.01 --d_shift 0.1 --mask_range 0.5 --confidence 0.    --lr_mult 2.   --low_degree .0
#
## reproduce baselines with L-Reg
#CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.0  --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.0  --d_shift 0.  --mask_range 0.5 --confidence 0.    --lr_mult 2.   --low_degree .5
#CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.0  --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.0  --d_shift 0.  --mask_range 0.5 --confidence 0.1   --lr_mult 2.   --low_degree .5
#CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.01 --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.0  --d_shift 0.  --mask_range 0.5 --confidence 0.    --lr_mult 2.   --low_degree .5
CUDA_VISIBLE_DEVICES=7 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO --ld 0.01 --tolerance_ratio 0.2 --model swag_regnety_16gf --batch_size 22 --swad True --shift 0.01 --d_shift 0.1 --mask_range 0.5 --confidence 0.    --lr_mult .25   --low_degree .1