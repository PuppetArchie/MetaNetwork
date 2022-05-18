# MetaNetwork


cd ../../../../

train_file="/mnt2/lzq/MetaNetwork/data/Cohn-Kanade-Database/train_pair_encode.txt"
test_file="/mnt2/lzq/MetaNetwork/data/Cohn-Kanade-Database/test_pair_encode.txt"

export CUDA_VISIBLE_DEVICES=4
python main/facial_expression_recognition/train_meta.py \
       --dataset="ck+" \
       --model="" \
       --checkpoint_dir="checkpoint" \
       --mode="generate" \
       --minor_mode="meta_classifier_hyper_attention" \
       --size="small" \
       --data_file="${train_file},${test_file}" \
       --batch_size=32 \
       --backbone="mobilenetv3"
