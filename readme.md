# Introduction
This repository contains instructions on how to fine tune a language model.

# Step 1: Setup Infra.

Press the button below for a one-click deployment to Azure using an ARM Template:

[![Deploy to Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fctava-msft%2Faistudio-finetuning%2Fmain%2Fazuredeploy.json)

# Step 2: Clone this repository

```
git clone https://github.com/ctava-msft/aistudio-finetuning
```

# Step 3: Setup Python environment

```
python -m venv .venv
pip install virtualenv
[windows].venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```
# Scripts

Make a file

```
python script.py
```

# Reference

[Fine tuning test generation](https://github.com/Azure/azureml-examples/blob/phi/bug_bash/sdk/python/foundation-models/system/finetune/text-generation/chat-completion.ipynb)
