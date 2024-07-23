from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import Input
from azure.identity import (
    DefaultAzureCredential,
    DeviceCodeCredential
)
import ast
import backtrace
import os
import time
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
AZURE_ENVIRONMENT = os.getenv("AZURE_ENVIRONMENT", "")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
ML_WORKSPACE_NAME = os.getenv("ML_WORKSPACE_NAME", "")
ML_EXPERIMENT_NAME = os.getenv("ML_EXPERIMENT_NAME", "")
ML_COMPONENT_NAME = os.getenv("ML_COMPONENT_NAME", "")
ML_COMPUTE_NAME = os.getenv("ML_COMPUTE_NAME", "")
ML_COMPUTE_SIZE = os.getenv("ML_COMPUTE_SIZE", "")
ML_DATASET_NAME = os.getenv("ML_DATASET_NAME", "")
ML_REGISTRY_NAME = os.getenv("ML_REGISTRY_NAME", "")
ML_FOUNDATION_MODEL_NAME = os.getenv("ML_FOUNDATION_MODEL_NAME", "")

# Obtain Default Azure Credentials
try:
    # Check if the environment is non-production
    is_non_production = AZURE_ENVIRONMENT != 'production'

    if is_non_production:
        # Use EnvironmentCredential for non-production
        credential = DeviceCodeCredential(tenant_id=AZURE_TENANT_ID)
    else:
        # Use DefaultAzureCredential for production
        credential = DefaultAzureCredential(tenant_id=AZURE_TENANT_ID)
        credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    backtrace.print_exc()
    raise ex


print(credential)
# Create the Workspace MLClient object
try:
    workspace_ml_client = MLClient.from_config(credential=credential)
except:
    workspace_ml_client = MLClient(
        credential,
        subscription_id=f"{AZURE_SUBSCRIPTION_ID}",
        resource_group_name=f"{AZURE_RESOURCE_GROUP}",
        workspace_name=f"{ML_WORKSPACE_NAME}",
    )
print(workspace_ml_client)

# Create the Registry MLClient object
registry_ml_client = MLClient(credential, registry_name=f"{ML_REGISTRY_NAME}")

# Get the Foundation Model
foundation_model = registry_ml_client.models.get(ML_FOUNDATION_MODEL_NAME, label="latest")
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for fine tuning".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)

# Download the Dataset
# This script will download the dataset and split it into train, validation and test sets
dataset_dir = "./dataset"
if not os.path.exists(dataset_dir):
    exit_status = os.system(
        f"python ./download-dataset.py --dataset {ML_DATASET_NAME} --download_dir {dataset_dir} --dataset_split_pc 5"
    )
    if exit_status != 0:
        raise Exception("Error downloading dataset")
else:
    print("Dataset already downloaded")

# Set Training Parameters
training_parameters = dict(
    #num_train_epochs=3,
    #per_device_train_batch_size=1,
    #per_device_eval_batch_size=1,
    #learning_rate=5e-6,
    #lr_scheduler_type="cosine",
)
print(f"The following training parameters are enabled - {training_parameters}")

# Get Optimization parameters
# These parameters are packaged with the model itself, lets retrieve those parameters
if "model_specific_defaults" in foundation_model.tags:
    optimization_parameters = ast.literal_eval(
        foundation_model.tags["model_specific_defaults"]
    )  # convert string to python dict
else:
    optimization_parameters = dict(
        apply_lora="true", apply_deepspeed="true", apply_ort="true", deepspeed_stage=3
    )
print(f"The following optimizations are enabled - {optimization_parameters}")

# Fetch the pipeline component
pipeline_component_func = registry_ml_client.components.get(
    name=f"{ML_COMPONENT_NAME}", label="latest"
)

# Define the pipeline job
@pipeline()
def create_pipeline():
    pipeline = pipeline_component_func(
        mlflow_model_path=foundation_model.id,
        compute_model_import=ML_COMPUTE_NAME,
        compute_preprocess=ML_COMPUTE_NAME,
        compute_finetune=ML_COMPUTE_NAME,
        # map the dataset splits to parameters
        train_file_path=Input(
            type="uri_file", path="./dataset/train_sft.jsonl"
        ),
        # validation_file_path=Input(
        #     type="uri_file", path="./dataset/validate.jsonl"
        # ),
        test_file_path=Input(
            type="uri_file", path="./dataset/test_sft.jsonl"
        ),
        # Training settings
        # set to the number of GPUs available in the compute
        number_of_gpu_to_use_finetuning=0,
        **training_parameters,
        **optimization_parameters
    )
    return {
        # map the output of the fine tuning job to the output of pipeline job so that we can easily register the fine tuned model
        # registering the model is required to deploy the model to an online or batch endpoint
        "trained_model": pipeline.outputs.mlflow_model_folder
    }

try:

    # get the workspace
    ws = workspace_ml_client.workspaces.get(f"{ML_WORKSPACE_NAME}")
    print(f"ws:{ws.location}-{ws.resource_group}")

    # create the pipeline
    pipeline_object = create_pipeline()

    # don't use cached results from previous jobs
    pipeline_object.settings.force_rerun = True

    # set continue on step failure to False
    pipeline_object.settings.continue_on_step_failure = False

    # submit the pipeline job
    pipeline_job = workspace_ml_client.jobs.create_or_update(
        pipeline_object, experiment_name=ML_EXPERIMENT_NAME
    )

    # wait for the pipeline to complete
    workspace_ml_client.jobs.stream(pipeline_job.name)
except Exception as ex:
    backtrace.print_exc()
    raise ex
