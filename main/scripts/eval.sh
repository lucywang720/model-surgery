export CUDA_VISIBLE_DEVICES=1

save_path=mistral
model_path=models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24

python -m eval.toxigen \
    --data_dir data/eval/toxigen/ \
    --save_dir results/$save_path/toxigen \
    --model_name_or_path $model_path \
    --eval_batch_size 1 \
    --max_prompts_per_group 500 \
    --use_vllm

python -m eval.realtoxicity \
    --data_dir data/eval/challenge_prompts.txt/ \
    --save_dir results/$save_path/toxicity \
    --model_name_or_path $model_path \
    --eval_batch_size 1 \
    --use_vllm

python -m eval.gsm \
    --data_dir data/eval/gsm/ \
    --max_num_examples 1319 \
    --save_dir results/$save_path/gsm \
    --model $model_path \
    --tokenizer $model_path \
    --n_shot 8 \
    --eval_batch_size 4 \
    --use_vllm

python -m eval.tydiqa \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_context_length 512 \
    --save_dir results/$save_path/tydiqa \
    --model /home/jxshi/mistral-test \
    --tokenizer /home/jxshi/mistral-test \
    --eval_batch_size 4 \
    --use_vllm \
    --max_num_examples_per_lang 100

python -m eval.bbh \
    --data_dir data/eval/bbh \
    --save_dir results/$save_path/bbh/ \
    --model $model_path \
    --tokenizer $model_path \
    --eval_batch_size 1 \
    --max_num_examples_per_task 40 \
    --use_vllm

python -m eval.mmlu \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/$save_path/mmlu \
    --model_name_or_path $model_path \
    --tokenizer_name_or_path $model_path \
    --eval_batch_size 1 \
    --load_in_8bit