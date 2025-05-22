# Exports
export PREFIX=eh-mega-cluster
export REGION=us-west1
export ZONE=us-west1-a
export PROJECT_ID=northam-ce-mlai-tpu
export GKE_VERSION=1.32.4-gke.1106000
export CLUSTER_NAME=eh-ray-enabled
export GSBUCKET=eh-ray
export PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
export NAMESPACE=default
export KSA_NAME=eh-ray
export HF_TOKEN=

# Configure networking
for N in $(seq 1 8); do
gcloud compute networks create ${PREFIX}-net-$N \
    --subnet-mode=custom \
    --mtu=8244

gcloud compute networks subnets create ${PREFIX}-sub-$N \
    --network=${PREFIX}-net-$N \
    --region=${REGION} \
    --range=192.167.$N.0/24

gcloud compute firewall-rules create ${PREFIX}-internal-$N \
  --network=${PREFIX}-net-$N \
  --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=192.167.0.0/16
done

# Provision a multi-node, ray-enabled gke cluster
gcloud container clusters create ${CLUSTER_NAME} \
    --region=${REGION} \
    --enable-ip-alias \
    --enable-dataplane-v2 \
    --cluster-version=${GKE_VERSION} \
    --node-locations ${ZONE} \
    --enable-multi-networking \
    --no-enable-autoupgrade \
    --machine-type=n2-standard-8 \
    --enable-ray-cluster-logging \
    --enable-ray-cluster-monitoring \
    --workload-pool=${PROJECT_ID}.svc.id.goog \
    --addons=RayOperator,GcsFuseCsiDriver

# Create network objects [MUST UPDATE network.yaml TO INCLUDE CORRECT PREFIX]
kubectl apply -f network.yaml

gcloud container node-pools create a3megax2nodes-multinic \
    --accelerator type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=latest \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=a3-megagpu-8g \
    --num-nodes=2 \
    --node-locations=${ZONE} \
    --no-enable-autoupgrade \
    --no-enable-autorepair \
    --additional-node-network network=${PREFIX}-net-1,subnetwork=${PREFIX}-sub-1 \
    --additional-node-network network=${PREFIX}-net-2,subnetwork=${PREFIX}-sub-2 \
    --additional-node-network network=${PREFIX}-net-3,subnetwork=${PREFIX}-sub-3 \
    --additional-node-network network=${PREFIX}-net-4,subnetwork=${PREFIX}-sub-4 \
    --additional-node-network network=${PREFIX}-net-5,subnetwork=${PREFIX}-sub-5 \
    --additional-node-network network=${PREFIX}-net-6,subnetwork=${PREFIX}-sub-6 \
    --additional-node-network network=${PREFIX}-net-7,subnetwork=${PREFIX}-sub-7 \
    --additional-node-network network=${PREFIX}-net-8,subnetwork=${PREFIX}-sub-8 \
    --enable-gvnic

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

# Deploy secret with HF token
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=${HF_TOKEN}

# Create TCPXO installer
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-tcpxo/nccl-tcpxo-installer.yaml

# Create NRI Device Injector Plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nri_device_injector/nri-device-injector.yaml

# Deploy NCCL test (Optional, but suggested.)
kubectl apply -f nccl-latest.yaml
kubectl exec --stdin --tty --container=nccl-test nccl-test-host-1 -- /scripts/allgather.sh nccl-host-1 nccl-host-2

# Clean Up NCCL test if completed
kubectl delete -f nccl-latest.yaml

# UPDATE ray-cluster-llama.yaml with SA and Bucket values
kubectl apply -f ray-cluster-config.yaml

# Submit finetune job.
# Get the head pod name
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head,ray.io/cluster=llama-raycluster -o jsonpath='{.items[0].metadata.name}')
echo "Head pod: $HEAD_POD"

# Port-forward (keep this running in a separate terminal)
kubectl port-forward $HEAD_POD 8265:8265

# Leverage virtual environment and install ray
python -m venv myenv && \
source myenv/bin/activate

pip install -U "ray[data,train,tune,serve]"

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