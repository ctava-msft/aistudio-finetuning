# Introduction
This repository contains instructions on how to fine tune a language model.
Language modeling has become ubiquitous for two main reasons:
- It's used to generate text
- LMs are exploitable. (thanks to transfer learning)
 
One can think in terms of layers. Where the bottom layers are trained on large amounts of text. This gives the model a vocabulary. These layers can be used directly as feature extractor, or they can be "fine-tuned" to provide focused context, content and reasoning. Steps listed below will create an Azure AI Studio environment and enable you to fine tune a model. Everything should be configurable in a .env file.

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

# Step 4: Review the Fine Tuning Program Design

- Get Default Credentials
- Create the ML Client
- Get the Foundation Model
- Get the Compute
- Download the Datasets
- Setup Training Properties
- Get Pipline Component(s)
- Create Pipeline
- Submit the Pipeline Job
- Wait for the Pipeline Job to be Completed

# Step 5: Kick off the fine tuning Job via a Python Script

If you are using a managed identity, AZURE_SUBSCRIPTION_ID and AZURE_RESOURCE_GROUP 
might not be neccessary.
Copy sample.env to .env
Enter variables information.

To execute the script, exxecute the following:
```
python script.py
```

# Reference(s)

[Fine tuning for text generation](https://github.com/Azure/azureml-examples/blob/phi/bug_bash/sdk/python/foundation-models/system/finetune/text-generation/chat-completion.ipynb)
