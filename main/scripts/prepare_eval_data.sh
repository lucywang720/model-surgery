mkdir -p data/downloads
mkdir -p data/eval

# MMLU dataset
wget -O data/downloads/mmlu_data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir -p data/downloads/mmlu_data
tar -xvf data/downloads/mmlu_data.tar -C data/downloads/mmlu_data
mv data/downloads/mmlu_data/data data/eval/mmlu && rm -r data/downloads/mmlu_data data/downloads/mmlu_data.tar


# Big-Bench-Hard dataset
wget -O data/downloads/bbh_data.zip https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip
mkdir -p data/downloads/bbh
unzip data/downloads/bbh_data.zip -d data/downloads/bbh
mv data/downloads/bbh/BIG-Bench-Hard-main/ data/eval/bbh && rm -r data/downloads/bbh data/downloads/bbh_data.zip

# TyDiQA-GoldP dataset
mkdir -p data/eval/tydiqa
wget -P data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json
wget -P data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json

# GSM dataset
wget -P data/eval/gsm/ https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl


# Multilingual GSM dataset
wget -O data/downloads/url-nlp.zip https://github.com/google-research/url-nlp/archive/refs/heads/main.zip
mkdir -p data/downloads/url-nlp
unzip data/downloads/url-nlp.zip -d data/downloads/url-nlp
mv data/downloads/url-nlp/url-nlp-main/mgsm data/eval/mgsm && rm -r data/downloads/url-nlp data/downloads/url-nlp.zip


# Codex HumanEval
wget -P data/eval/codex_humaneval https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz


# TruthfulQA
wget -P data/eval/truthfulqa https://github.com/sylinrl/TruthfulQA/raw/main/TruthfulQA.csv

# Toxigen data
mkdir -p data/eval/toxigen
for minority_group in asian black chinese jewish latino lgbtq mental_disability mexican middle_east muslim native_american physical_disability trans women
do
    wget -O data/eval/toxigen/hate_${minority_group}.txt https://raw.githubusercontent.com/microsoft/TOXIGEN/main/prompts/hate_${minority_group}_1k.txt
done
