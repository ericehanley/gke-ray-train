export REGION=us-west1
export ZONE=us-west1-a
export PROJECT_ID=northam-ce-mlai-tpu
export GKE_VERSION=1.32.2-gke.1297002
export CLUSTER_NAME=eh-ray-enabled
export GSBUCKET=eh-ray
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
export NAMESPACE=default
export KSA_NAME=eh-ray
export NUM_NODES=2
export NUM_GPUS_PER_NODE=8
export HF_TOKEN=

# Provision a single node, ray-enabled gke cluster
gcloud container clusters create ${CLUSTER_NAME} \
    --region=${REGION} \
    --node-locations=${ZONE} \
    --cluster-version=${GKE_VERSION} \
    --machine-type=n2-standard-8 \
    --num-nodes=1 \
    --enable-ray-cluster-logging \
    --enable-ray-cluster-monitoring \
    --workload-pool=${PROJECT_ID}.svc.id.goog \
    --addons=RayOperator,GcsFuseCsiDriver

gcloud container node-pools create a3megax2nodes \
    --accelerator type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=latest \
    --node-version=${GKE_VERSION} \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --node-locations=${ZONE} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=a3-megagpu-8g \
    --num-nodes=${NUM_NODES}

# Leverage virtual environment and install ray
python -m venv myenv && \
source myenv/bin/activate

pip install -U "ray[data,train,tune,serve]"

# Create Cloud Storage Bucket & Configure
gcloud storage buckets create gs://${GSBUCKET} \
    --uniform-bucket-level-access \
    --location=${REGION}\
    --enable-hierarchical-namespace

# Create GKE SA
kubectl create serviceaccount ${KSA_NAME}

# Add permissions to bucket for FUSE CSI driver
gcloud storage buckets add-iam-policy-binding gs://${GSBUCKET} \
  --member "principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/${NAMESPACE}/sa/${KSA_NAME}" \
  --role "roles/storage.objectUser"

# Deploy RayCluster
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=${HF_TOKEN}

# UPDATE ray-cluster-llama.yaml with SA and Bucket values and deploy
envsubst < a3-mega/ray-cluster-config.yaml | kubectl apply -f -

# Submit finetune job.
# Get the head pod name
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head,ray.io/cluster=llama-raycluster -o jsonpath='{.items[0].metadata.name}')
echo "Head pod: $HEAD_POD"

# Port-forward (keep this running in a separate terminal)
kubectl port-forward $HEAD_POD 8265:8265

# From separate terminal in same directory as your script
ray job submit --address http://localhost:8265 --runtime-env-json='{
    "working_dir": ".",
    "pip": [
        "torch==2.3.0",
        "datasets==3.6.0",
        "transformers==4.50.0",
        "peft==0.15.0",
        "accelerate==1.6.0",
        "trl==0.17.0",
        "bitsandbytes==0.45.0"
    ],
    "env_vars": {
        "HF_TOKEN": "'"$HF_TOKEN"'",
        "NUM_NODES": "'"$NUM_NODES"'",
        "NUM_GPUS_PER_NODE": "'"$NUM_GPUS_PER_NODE"'"
    }
}' -- python ray-jobs/fine_tune_llama_ray.py