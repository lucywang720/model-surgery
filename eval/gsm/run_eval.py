import argparse
import os
import re
import json
import random
import torch
import vllm
import evaluate
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    dynamic_import_function,
)
from eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"


exact_match = evaluate.load("metrics/exact_match")

def truncate_after_string(original_string, target_string):
    position = original_string.find(target_string)
    if position != -1:  # 如果找到了目标字符串
        return original_string[:position]
    else:  # 如果没有找到目标字符串
        return original_string



def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    print(args.data_dir)
    with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })
        
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
        

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    global GSM_EXAMPLARS
    if args.n_shot:
        if len(GSM_EXAMPLARS) > args.n_shot:
            GSM_EXAMPLARS = random.sample(GSM_EXAMPLARS, args.n_shot)
        demonstrations = []
        for example in GSM_EXAMPLARS:
            if args.no_cot:
                demonstrations.append(
                    "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                )
            else:
                demonstrations.append(
                    "Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"]
                )
        prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
    else:
        for example in GSM_EXAMPLARS:
            # print(example)
            prompt_prefix = "Answer the following question.\n\n" + example['question']
            break


    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in test_data:
        prompt = prompt_prefix + "Question: " + example["question"].strip()
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "Answer:"
            else:
                prompt += " Answer:"
        else:
            prompt += "\nAnswer:"
        prompts.append(prompt)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        if args.use_vllm:
            # tokenizer.add_tokens(["[s1]", "[s2]"])
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
                tokenizer_mode="slow" if args.use_slow_tokenizer else "auto",
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=0.75,
                max_num_seqs=4
            )
            # if torch.cuda.device_count() != 1:
                # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            # model.module.get_tokenizer().add_tokens(["[s1]", "[s2]"])
            # model.get_tokenizer().add_tokens(["[s1]", "[s2]"])
            sampling_params = vllm.SamplingParams(
                temperature=0,
                max_tokens=256,
                stop=["\n"] if not args.use_chat_format else None,
            )
            # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
            # if torch.cuda.device_count() != 1:
                # generations = model.module.generate(prompts, sampling_params)
            # else:
            generations = model.generate(prompts, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [truncate_after_string(prompt_to_output[prompt], '<|endoftext|>') if prompt in prompt_to_output else "" for prompt in prompts]
        else:
            print("?")
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path, 
                tokenizer_name_or_path=args.tokenizer_name_or_path, 
                load_in_8bit=args.load_in_8bit, 
                device_map="auto",
                gptq_model=args.gptq,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=256,
                batch_size=args.eval_batch_size,
                stop_id_sequences=[[new_line_token]],
                do_sample=False,
            )
    else:
        instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
        results = query_openai_chat_model(
            engine=args.openai_engine,
            instances=instances,
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=os.path.join(args.save_dir, f"openai_results.jsonl"),
        )
        outputs = [result["output"] for result in results]

    predictions = []
    for output in outputs:
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)
        
    print("Calculating accuracy...")
    targets = [example["answer"] for example in test_data]

    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, outputs, predictions)]

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/eval/gsm/"
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--n_shot", 
        type=int, 
        default=8, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="If given, we're evaluating a model without chain-of-thought."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true", 
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
