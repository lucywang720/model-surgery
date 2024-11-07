export CUDA_VISIBLE_DEVICES=0

model_path=model_path
save_path=save_path
model_name=model_name
alpha=1.08
toxic_probe_path=toxic_epoch_9.pt

python -m modification.modify \
    --data_dir data/eval/toxigen/ \
    --save_dir $save_path \
    --model_name_or_path llama2-7b-hf  \
    --eval_batch_size 1 \
    --alpha $alpha \
    --toxic_path $toxic_probe_path \
    --max_prompts_per_group 10 \
    --save_model

python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_context_length 512 \
    --save_dir results/tydiqa/$save_path \
    --model $model_path \
    --tokenizer $model_path \
    --eval_batch_size 4 \
    --use_vllm

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --max_num_examples 1319 \
    --save_dir results/gsm/$save_path \
    --model $model_path \
    --tokenizer $model_path \
    --n_shot 8 \
    --use_vllm

python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/$save_path/ \
    --model $model_path \
    --tokenizer $model_path \
    --eval_batch_size 1 \
    --max_num_examples_per_task 40 \
    --use_vllm

python -m eval.toxigen.run_eval_sub \
    --data_dir data/eval/toxigen/ \
    --save_dir results/toxigen/$save_path/ \
    --model_name_or_path $model_path \
    --eval_batch_size 1 \
    --use_vllm


python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir results/mmlu/$save_path \
    --model_name_or_path $model_path  \
    --tokenizer_name_or_path $model_path  \
    --eval_batch_size 1

