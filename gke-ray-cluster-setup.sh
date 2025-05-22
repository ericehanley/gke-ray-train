# Provision a multi-node, ray-enabled gke cluster
gcloud container clusters create ${CLUSTER_NAME} \
    --zone=${ZONE} \
    --cluster-version=1.32.4-gke.1106000 \
    --machine-type=n2-standard-8 \
    --enable-ray-cluster-logging \
    --enable-ray-cluster-monitoring \
    --workload-pool=${PROJECT_ID}.svc.id.goog \
    --addons=RayOperator,GcsFuseCsiDriver

gcloud container node-pools create a3megax2nodes \
    --accelerator type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=latest \
    --project=${PROJECT_ID} \
    --location=${ZONE} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=a3-megagpu-8g \
    --num-nodes=1

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

# UPDATE ray-cluster-llama.yaml with SA and Bucket values
kubectl apply -f ray-cluster-config.yaml

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
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "HF_TOKEN": "'"$HF_TOKEN"'"
    }
}' -- python src/fine_tune_llama_ray.py