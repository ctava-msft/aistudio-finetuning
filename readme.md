# Introduction
This set of scripts that is used to fine tune a language model.

# Setup Infra.

[![Deploy to Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fctava-msft%2Faistudio-finetuning%2Fmain%2Fazuredeploy.json)


# Setup environment
```
python -m venv .venv
pip install virtualenv
[windows].venv\Scripts\activate
[mac]activate
pip install -r requirements.txt
```
# Scripts

```
python script.py
```

# Reference

[Fine tuning test generation](https://github.com/Azure/azureml-examples/blob/phi/bug_bash/sdk/python/foundation-models/system/finetune/text-generation/chat-completion.ipynb)
