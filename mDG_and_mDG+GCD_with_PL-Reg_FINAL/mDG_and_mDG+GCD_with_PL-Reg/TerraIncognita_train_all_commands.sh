DATASET="TerraIncognita"
DATADIR="/data1/phd21_zhaorui_tan/data_raw/DG"
#DATADIR="/Data_PHD/phd22_zhaorui_tan/data_raw/dg_data/DG"


# reproduce baselines, including ERM, CE, MIRO, GMDG
#CUDA_VISIBLE_DEVICES=1 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.0   --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree 0.
#CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.0   --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.1   --lr_mult 10.  --low_degree 0.
#CUDA_VISIBLE_DEVICES=1 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree 0.
#CUDA_VISIBLE_DEVICES=1 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.01 --d_shift 0.01  --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree 0.

# reproduce baselines with L-Reg
#CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.0   --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree .5
#CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.0   --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.1   --lr_mult 10.  --low_degree .5
#CUDA_VISIBLE_DEVICES=4 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree .5

#CUDA_VISIBLE_DEVICES=7 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.001 --d_shift 0.001  --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree .5
#CUDA_VISIBLE_DEVICES=5 python train_all_mdl.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.   --d_shift 0.    --mask_range 0.5 --confidence 0.    --lr_mult 10.  --low_degree 1.
CUDA_VISIBLE_DEVICES=0 python train_all.py exp_name --data_dir ${DATADIR}  --dataset ${DATASET}  --algorithm MIRO  --model swag_regnety_16gf --batch_size 22 --swad True  --ld 0.01  --shift 0.01 --d_shift 0.01  --mask_range 0.5 --confidence 0.    --lr_mult 1.  --low_degree .5
