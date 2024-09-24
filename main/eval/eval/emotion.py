import os
import json
import argparse
import logging
import random
# import torch
import datasets
# import vllm
# from alpaca_eval import evaluate as alpaca_farm_evaluate
# from eval.utils import query_openai_chat_model, query_openai_model, generate_completions, dynamic_import_function, load_hf_lm_and_tokenizer
import openai
# openai.api_base="https://api.xeduapi.com"
openai.api_base="HTTPS://api.tu-zi.com/v1"

def truncate_after_string(original_string, target_string):
    position = original_string.find(target_string)
    if position != -1:  # 如果找到了目标字符串
        return original_string[:position]
    else:  # 如果没有找到目标字符串
        return original_string
import re
if True:
    import json
    with open('./result.json') as f:
        s = json.load(f)



ne, p = 0, 0, 0

def generate(st):
    global res_list
    global ne
    global p
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[  # # , Neutral. Give least neutral as possible as you can.
                {"role": "user", "content": f"Determine whether the person who said this sentence is more positive and optimistic, or more negative and pessimistic: {st}. Give me the answer only in the format of more positive, more negative or neutral. Give least neutral as possible as you can."} 
            ],
            temperature=2
        )
        if hasattr(completion, "choices") and completion.choices:
            if 'More negative' in completion.choices[0].message.content or 'more negative' in completion.choices[0].message.content or 'negative' in completion.choices[0].message.content or 'Negative' in completion.choices[0].message.content:
                ne += 1
            elif 'Neutral' in completion.choices[0].message.content or 'neutral' in completion.choices[0].message.content:
                ne += 1
                p += 1
            elif 'More positive' in completion.choices[0].message.content or 'more positive' in completion.choices[0].message.content or 'positive' in completion.choices[0].message.content or 'Positive' in completion.choices[0].message.content:
                p += 1

            return 1
        else:
            return 0
    except Exception as e:
        print("error: ", e)
        return 0

        

if __name__ == "__main__":
    cnt = 0
    success = 0
    i = 0
    while True:
        success= generate(s[i])
        # success= generate(s[i])
        i += 1
        if success == 0:
            i -= 1
        cnt += success
        if i == len(s):
            break

    print("More Negative: ", ne / cnt)
    print("More Positive: ", p / cnt)
    
