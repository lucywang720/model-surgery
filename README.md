# Model Surgery: Modulating LLM's Behavior Via Simple Parameter Editing

official code of the following paper:

[Model Surgery: Modulating LLM's Behavior Via Simple Parameter Editing](https://arxiv.org/abs/2407.08770)

$^1$ Department of Automation, BNRist, Tsinghua University $^2$ Carnegie Mellon University.

![main](/pic/main.png)

## Main Results

Large Language Models (LLMs) have shown great potential as AI assistants, but ensuring their safety and reliability remains a challenge. Current methods for aligning LLM behavior, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), are computationally expensive and may degrade model performance.

We introduce a novel approach called "model surgery" that modulates LLM behavior by directly editing a small subset of model parameters. Our method:

- [x] Uses a linear classifier (behavior probe) to identify critical parameters influencing targeted behaviors

- [x] Edits selected parameters to shift model outputs away from undesirable behaviors

- [x] Requires only inference-level computational resources

- [x] Preserves core model capabilities while effectively modulating behavior

We demonstrate the effectiveness of our approach on tasks including detoxification, jailbreak resistance, and attitude adjustment.

![toxic](/pic/toxic.png)

One example of our method: 

![example](/pic/example.png)

## Setup

#### Model downloading

| Model           | Download                                             |
| --------------- | ---------------------------------------------------- |
| LLaMA2-7B       | https://huggingface.co/meta-llama/Llama-2-7b-hf      |
| LLaMA2-7B-Chat  | https://huggingface.co/meta-llama/Llama-2-7b-chat-hf |
| CodeLLaMA-7B    | https://huggingface.co/meta-llama/CodeLlama-7b-hf    |
| Mistral-7B-v0.1 | https://huggingface.co/mistralai/Mistral-7B-v0.1     |

#### Data Preparation

```python
bash scripts/prepare_eval_data.sh
```

for other data, you can find them at [here](https://drive.google.com/drive/folders/1yDqVOEdC7E66XwtODvA1rN0zxJFuwFCd?dmr=1&ec=wgc-drive-globalnav-goto)

#### Pip installation

```python
git clone https://github.com/lucywang720/model-surgery.git
cd model-surgery
conda create -n sugery python=3.9
conda activate sugery
pip install -r requirements.txt
```



## Training & Evaluation Steps

We offer the scripts to directly run our experiments

#### Behavior Probe Extraction

This step trains a linear classifier to identify specific behaviors in the LLM's hidden states.

```python
bash scripts/training.sh
```

or 
```python
python -m train --data_path jigsaw.txt --save_model --pretrained_model llama2 --epochs 20 --learning_rate 0.0001 --output_fp probe_llama --batch_size 32
```


#### Model Surgery

Using the extracted probe, this step modifies selected model parameters to shift behavior. You may need to add your own probe path to the scripts.

```python
bash scripts/modify.sh
```
or 
```python
python -m modify \
    --save_dir llama2-non-toxic \
    --model_name_or_path llama2  \
    --alpha $alpha \
    --toxic_path probe.pt \
    --save_model
```

#### Evaluation

Assess the performance of the modified model on various tasks to ensure behavior change and capability preservation. We offer one-click running scripts.

```python
bash scripts/eval.sh
```

#### Released Checkpoints

For quick experimentation, you can use our pre-trained behavior probes offered in ./main/modification/checkpoint.

## Citation

```python
@misc{wang2024modelsurgerymodulatingllms,
      title={Model Surgery: Modulating LLM's Behavior Via Simple Parameter Editing}, 
      author={Huanqian Wang and Yang Yue and Rui Lu and Jingxin Shi and Andrew Zhao and Shenzhi Wang and Shiji Song and Gao Huang},
      year={2024},
      eprint={2407.08770},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.08770}, 
}
```

## Contact

The code in this repository is still being reorganized. Errors that may arise during the organizing process could lead to code malfunctions or discrepancies from the original research results. If you encounter any problems, please raise issues. I will go and fix these bugs.
For any questions or feedback, please open an issue or contact the author: wang-hq23@mails.tsinghua.edu.cn
