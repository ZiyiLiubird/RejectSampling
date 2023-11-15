
# for VARIABLE in 0 1 2 3 4 5 6 7
# do
# CUDA_VISIBLE_DEVICES=$VARIABLE nohup python -u sample_main.py --rank $VARIABLE \
#                                     --data_path /alg_vepfs/public/datasets/joyland/7days/3736sample/7days3k_$VARIABLE.json \
#                                     --sample_k 5 &> p$VARIABLE.out &
# done

CUDA_VISIBLE_DEVICES=7 nohup python -u sample_main.py --rank 7 \
                                    --data_path /alg_vepfs/public/datasets/joyland/7days/3736sample/7days3k_7.json \
                                    --sample_k 5 &> p7.out &
