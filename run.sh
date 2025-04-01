# flickr
python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py \
--config ./configs/Retrieval_flickr.yaml \
--output_dir output/Retrieval_flickr \
--checkpoint [Pretrained checkpoint]

# coco: 8*A100
python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py \
--config ./configs/Retrieval_coco_224.yaml \
--output_dir /exp/output/FD/ALBEF_Retrieval_coco_224 \
--checkpoint /exp/pretrain_models/ALBEF_4M.pth

# local 12G xp: 224; bs 16: 9G; 2h/ep -> add ckpt: 224; bs 32: 9G; 2h/ep
# -> 384 + bs 32: will oom 32G?; add ckpt will ok -> 1.5h/ep 8G V100; 224: 0.5h

# set up
# 0. git clone https://github.com/weiyx16/ALBEF.git
# 1. download data.tar.gz
# wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/data.tar.gz; tar -xf data.tar.gz
# 2. azcopy cocoir;
# 3. training