Due to the limited space, only two loras are uploaded for model parameters;
You need to download codellama-7B-Instruct-hf from huggingface;
Then use code/train/script/merge_lora.py to merge model/newbase in turn to get new_base_merged, and then merge new_base_merged with newbase_tuned_on_baseline again to get the fine-tuned version on the baseline dataset

train code's run way: code/train/run_train.sh (You need to download codellama-7B-Instruct-hf)
test code's run way: code/test/test_baseline.py  or  code/test/test_ourdataset.py  (You need to download codellama-7B-Instruct-hf, then merge lora model in ./model; or you have finished training)
