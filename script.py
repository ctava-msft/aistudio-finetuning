from azure.ai.ml import MLClient
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, PipelineComponent, Job, Component
from azure.ai.ml import PyTorchDistribution, Input
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)
from azure.ai.ml.entities import AmlCompute
import ast
import os
import time
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")
ML_WORKSPACE_NAME = os.getenv("ML_WORKSPACE_NAME", "")
ML_EXPERIMENT_NAME = os.getenv("ML_EXPERIMENT_NAME", "")
ML_COMPUTE_NAME = os.getenv("ML_COMPUTE_NAME", "")
ML_COMPUTE_SIZE = os.getenv("ML_COMPUTE_SIZE", "")
ML_DATASET_NAME = os.getenv("ML_DATASET_NAME", "")
ML_REGISTRY_NAME = os.getenv("ML_REGISTRY_NAME", "")
ML_FOUNDATION_MODEL_NAME = os.getenv("ML_FOUNDATION_MODEL_NAME", "")

# Obtain Default Azure Credentials
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

# Create the Workspace MLClient object
try:
    workspace_ml_client = MLClient.from_config(credential=credential)
except:
    workspace_ml_client = MLClient(
        credential,
        subscription_id=f"{AZURE_SUBSCRIPTION_ID}",
        resource_group_name=f"{AZURE_RESOURCE_GROUP}",
        workspace_name=f"{AZURE_RESOURCE_GROUP}",
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

# Get the compute
try:
    compute = workspace_ml_client.compute.get(ML_COMPUTE_NAME)
    print("The compute cluster already exists! Reusing it for the current run")
except Exception as ex:
    print(
        f"Looks like the compute cluster doesn't exist. Creating a new one with compute size {ML_COMPUTE_SIZE}!"
    )
    try:
        print("Attempt #1 - Trying to create a dedicated compute")
        compute = AmlCompute(
            name=ML_COMPUTE_NAME,
            size=ML_COMPUTE_SIZE,
            tier="Dedicated",
            max_instances=2,  # For multi node training set this to an integer value more than 1
        )
        workspace_ml_client.compute.begin_create_or_update(compute).wait()
    except Exception as e:
        try:
            print(
                "Attempt #2 - Trying to create a low priority compute. Since this is a low priority compute, the job could get pre-empted before completion."
            )
            compute = AmlCompute(
                name=ML_COMPUTE_NAME,
                size=ML_COMPUTE_SIZE,
                tier="LowPriority",
                max_instances=2,  # For multi node training set this to an integer value more than 1
            )
            workspace_ml_client.compute.begin_create_or_update(compute).wait()
        except Exception as e:
            print(e)
            raise ValueError(
                f"WARNING! Compute size {ML_COMPUTE_SIZE} not available in workspace"
            )


# Sanity check on the created compute
compute = workspace_ml_client.compute.get(ML_COMPUTE_NAME)
if compute.provisioning_state.lower() == "failed":
    raise ValueError(
        f"Provisioning failed, Compute '{ML_COMPUTE_NAME}' is in failed state. "
        f"please try creating a different compute"
    )



# This is the number of GPUs in a single node of the selected 'vm_size' compute.
# Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
# Setting this to more than the number of GPUs will result in an error.
gpu_count_found = False
workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
available_sku_sizes = []
for compute_sku in workspace_compute_sku_list:
    available_sku_sizes.append(compute_sku.name)
    if compute_sku.name.lower() == compute.size.lower():
        gpus_per_node = compute_sku.gpus
        gpu_count_found = True
# if gpu_count_found not found, then print an error
if gpu_count_found:
    print(f"Number of GPU's in compute {compute.size}: {gpus_per_node}")
else:
    raise ValueError(
        f"Number of GPU's in compute {compute.size} not found. Available skus are: {available_sku_sizes}."
        f"This should not happen. Please check the selected compute cluster: {ML_COMPUTE_NAME} and try again."
    )

# Download the Dataset
# This script will download the dataset and split it into train, validation and test sets
exit_status = os.system(
    f"python ./download-dataset.py --dataset {ML_DATASET_NAME} --download_dir dataset --dataset_split_pc 5"
)
if exit_status != 0:
    raise Exception("Error downloading dataset")

# Set Training Parameters
training_parameters = dict(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
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
    name="chat_completion_pipeline", label="latest"
)

# Define the pipeline job
@pipeline()
def create_pipeline():
    chat_completion_pipeline = pipeline_component_func(
        mlflow_model_path=foundation_model.id,
        compute_model_import=ML_COMPUTE_NAME,
        compute_preprocess=ML_COMPUTE_NAME,
        compute_finetune=ML_COMPUTE_NAME,
        # map the dataset splits to parameters
        train_file_path=Input(
            type="uri_file", path="./dataset/train.jsonl"
        ),
        validation_file_path=Input(
            type="uri_file", path="./dataset/validate.jsonl"
        ),
        test_file_path=Input(
            type="uri_file", path="./dataset/test.jsonl"
        ),
        # Training settings
        # set to the number of GPUs available in the compute
        number_of_gpu_to_use_finetuning=gpus_per_node,
        **training_parameters,
        **optimization_parameters
    )
    return {
        # map the output of the fine tuning job to the output of pipeline job so that we can easily register the fine tuned model
        # registering the model is required to deploy the model to an online or batch endpoint
        "trained_model": chat_completion_pipeline.outputs.mlflow_model_folder
    }

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
