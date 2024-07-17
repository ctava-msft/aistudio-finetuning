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
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", "")

# Obtain Default Azure Credentials
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

# Create the MLClient object
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
'''
# the models, fine tuning pipelines and environments are available in the AzureML system registry, "azureml"
registry_ml_client = MLClient(credential, registry_name="azureml")
registry_ml_client_staging = MLClient(credential, registry_name="models-staging")
registry_ml_client_msr = MLClient(credential, registry_name="azureml-msr")
registry_ml_client_meta = MLClient(credential, registry_name="azureml-meta")
experiment_name = "chat-completion"

# generating a unique timestamp that can be used for names and versions that need to be unique
timestamp = str(int(time.time()))

model_name = "Phi-3-mini-128k-instruct"
foundation_model = registry_ml_client_staging.models.get(model_name, label="latest")
# foundation_model = registry_ml_client_msr.models.get(model_name, label="latest")
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for fine tuning".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)

if "computes_allow_list" in foundation_model.tags:
    computes_allow_list = ast.literal_eval(
        foundation_model.tags["computes_allow_list"]
    )  # convert string to python list
    print(f"Please create a compute from the above list - {computes_allow_list}")
else:
    computes_allow_list = None
    print("Computes allow list is not part of model tags")

# If you have a specific compute size to work with change it here. By default we use the 8 x V100 compute from the above list
compute_cluster_size = "Standard_ND40rs_v2"

# If you already have a gpu cluster, mention it here. Else will create a new one with the name 'gpu-cluster-big'
compute_cluster = "gpu-cluster-big"

try:
    compute = workspace_ml_client.compute.get(compute_cluster)
    print("The compute cluster already exists! Reusing it for the current run")
except Exception as ex:
    print(
        f"Looks like the compute cluster doesn't exist. Creating a new one with compute size {compute_cluster_size}!"
    )
    try:
        print("Attempt #1 - Trying to create a dedicated compute")
        compute = AmlCompute(
            name=compute_cluster,
            size=compute_cluster_size,
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
                name=compute_cluster,
                size=compute_cluster_size,
                tier="LowPriority",
                max_instances=2,  # For multi node training set this to an integer value more than 1
            )
            workspace_ml_client.compute.begin_create_or_update(compute).wait()
        except Exception as e:
            print(e)
            raise ValueError(
                f"WARNING! Compute size {compute_cluster_size} not available in workspace"
            )


# Sanity check on the created compute
compute = workspace_ml_client.compute.get(compute_cluster)
if compute.provisioning_state.lower() == "failed":
    raise ValueError(
        f"Provisioning failed, Compute '{compute_cluster}' is in failed state. "
        f"please try creating a different compute"
    )

if computes_allow_list is not None:
    computes_allow_list_lower_case = [x.lower() for x in computes_allow_list]
    if compute.size.lower() not in computes_allow_list_lower_case:
        raise ValueError(
            f"VM size {compute.size} is not in the allow-listed computes for finetuning"
        )
else:
    # Computes with K80 GPUs are not supported
    unsupported_gpu_vm_list = [
        "standard_nc6",
        "standard_nc12",
        "standard_nc24",
        "standard_nc24r",
    ]
    if compute.size.lower() in unsupported_gpu_vm_list:
        raise ValueError(
            f"VM size {compute.size} is currently not supported for finetuning"
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
        f"This should not happen. Please check the selected compute cluster: {compute_cluster} and try again."
    )

# download the dataset using the helper script. This script will download the dataset and split it into train, validation and test sets
exit_status = os.system(
    "python ./download-dataset.py --dataset HuggingFaceH4/ultrachat_200k --download_dir dataset --dataset_split_pc 5"
)
if exit_status != 0:
    raise Exception("Error downloading dataset")

# Training parameters
training_parameters = dict(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
)
print(f"The following training parameters are enabled - {training_parameters}")

# Optimization parameters - As these parameters are packaged with the model itself, lets retrieve those parameters
if "model_specific_defaults" in foundation_model.tags:
    optimization_parameters = ast.literal_eval(
        foundation_model.tags["model_specific_defaults"]
    )  # convert string to python dict
else:
    optimization_parameters = dict(
        apply_lora="true", apply_deepspeed="true", apply_ort="true", deepspeed_stage=3
    )
print(f"The following optimizations are enabled - {optimization_parameters}")

# fetch the pipeline component
pipeline_component_func = registry_ml_client_staging.components.get(
    name="chat_completion_pipeline", label="latest"
)

# define the pipeline job
@pipeline()
def create_pipeline():
    chat_completion_pipeline = pipeline_component_func(
        mlflow_model_path=foundation_model.id,
        compute_model_import=compute_cluster,
        compute_preprocess=compute_cluster,
        compute_finetune=compute_cluster,
        # map the dataset splits to parameters
        train_file_path=Input(
            type="uri_file", path="./dataset/train_sft.jsonl"
        ),
        # validation_file_path=Input(
        #     type="uri_file", path="./dataset/test_.jsonl"
        # ),
        test_file_path=Input(type="uri_file", path="./dataset/test_sft.jsonl"),
        # Training settings
        number_of_gpu_to_use_finetuning=gpus_per_node,  # set to the number of GPUs available in the compute
        **training_parameters,
        **optimization_parameters
    )
    return {
        # map the output of the fine tuning job to the output of pipeline job so that we can easily register the fine tuned model
        # registering the model is required to deploy the model to an online or batch endpoint
        "trained_model": chat_completion_pipeline.outputs.mlflow_model_folder
    }


pipeline_object = create_pipeline()

# don't use cached results from previous jobs
pipeline_object.settings.force_rerun = True

# set continue on step failure to False
pipeline_object.settings.continue_on_step_failure = False

# submit the pipeline job
pipeline_job = workspace_ml_client.jobs.create_or_update(
    pipeline_object, experiment_name=experiment_name
)
# wait for the pipeline job to complete
workspace_ml_client.jobs.stream(pipeline_job.name)

'''