python -m modification.modify \
    --save_path save_path \
    --model_name_or_path models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24 \
    --alpha $alpha \
    --save_model \
    --toxic_path modification/checkpoint/mistral.pt