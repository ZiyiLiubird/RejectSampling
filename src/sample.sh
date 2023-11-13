# CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --sample_num 15000 --sample_k 5 &> p0.out &
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --sample_num 15000 --sample_k 5 &> p1.out &
# python -u main.py --sample_num 10 --sample_k 2