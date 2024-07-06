# Model Surgery: Modulating LLM's Behavior Via Simple Parameter Editing

official code of the following paper:

link

$^1$ Department of Automation, BNRist, Tsinghua University $^2$ Carnegie Mellon University.

[pic]

## Main Results

Large Language Models (LLMs) have shown great potential as AI assistants, but ensuring their safety and reliability remains a challenge. Current methods for aligning LLM behavior, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), are computationally expensive and can degrade model performance.

We introduce a novel approach called "model surgery" that modulates LLM behavior by directly editing a small subset of model parameters. Our method:

- [x] Uses a linear classifier (behavior probe) to identify critical parameters influencing targeted behaviors

- [x] Edits selected parameters to shift model outputs away from undesirable behaviors

- [x] Requires only inference-level computational resources

- [x] Preserves core model capabilities while effectively modulating behavior

We demonstrate the effectiveness of our approach on tasks including detoxification, jailbreak resistance, and attitude adjustment.

[pic]

## Setup

#### Model downloading

LLaMA2-7B git clone https://huggingface.co/meta-llama/Llama-2-7b-hf

LLaMA2-7B-Chat git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

CodeLLaMA-7B git clone https://huggingface.co/meta-llama/CodeLlama-7b-hf

Mistral-7B-v0.1 git clone https://huggingface.co/mistralai/Mistral-7B-v0.1

#### Data Preparation

```python
bash download_data.sh
```

#### Pip installation

```python
git clone ...
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

#### Model Surgery

Using the extracted probe, this step modifies selected model parameters to shift behavior. You may need to add your own probe path to the scripts.

```python
bash scripts/modify.sh
```

#### Evaluation

Assess the performance of the modified model on various tasks to ensure behavior change and capability preservation.

```python
bash scripts/eval.sh
```

#### Released Checkpoints

For quick experimentation, you can use our pre-trained behavior probes offered in ./checkpoint.

## Citation



## Contact

For any questions or feedback, please open an issue or contact the author: wang-hq23@mails.tsinghua.edu.cn
