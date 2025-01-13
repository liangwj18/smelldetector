export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node=4 train.py --train_args_file ./train_args/sft/detector_qlora/codellama-7b-sft-qlora.json
