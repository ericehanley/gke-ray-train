# Provision a multi-node, ray-enabled gke cluster
gcloud container clusters create ${CLUSTER_NAME} \
    --zone=${ZONE} \
    --release-channel=rapid \
    --machine-type=n2-standard-8 \
    --enable-ray-cluster-logging \
    --enable-ray-cluster-monitoring \
    --addons=RayOperator

gcloud container node-pools create a3megax2nodes \
    --accelerator type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=latest \
    --project=${PROJECT_ID} \
    --location=${ZONE} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=a3-megagpu-8g \
    --num-nodes=2

# Leverage virtual environment and install ray
python -m venv myenv && \
source myenv/bin/activate

pip install -U "ray[data,train,tune,serve]"

# KubeRay Operator Installation: https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/kuberay-operator-installation.html
"""
This is a Kubernetes operator, which is essentially a custom controller. Its job is to extend Kubernetes' capabilities to understand and manage Ray applications.
The KubeRay operator watches for specific Ray-related custom resources (like RayCluster, RayJob, and RayService) and takes action to create, configure, and manage the underlying Kubernetes resources (like Pods and Services) needed to run a Ray cluster.
It automates the deployment, scaling, and lifecycle management of Ray clusters on Kubernetes.
"""
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
# Install both CRDs and KubeRay operator v1.2.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.2.0

# Validate kuberay operator is running (optional)
kubectl get pods

# Create image for RayCluster
gcloud artifacts repositories create docker-repo \
    --repository-format=docker \
    --location=${GCP_REGION} \
    --description="Docker repository for Ray images"

gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

export IMAGE_NAME="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/docker-repo/ray-llama-finetune:latest"
docker build -t "${IMAGE_NAME}" .
docker push "${IMAGE_NAME}"

# Deploy RayCluster
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=${HF_TOKEN}
kubectl apply -f ray-cluster-llama.yaml

# Check the status:
kubectl get rayclusters
kubectl get pods -l ray.io/cluster=llama-raycluster
# Wait until head and worker pods are in 'Running' state.

# Submit finetune job.
# Get the head pod name
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head,ray.io/cluster=llama-raycluster -o jsonpath='{.items[0].metadata.name}')
echo "Head pod: $HEAD_POD"

# Port-forward (keep this running in a separate terminal)
kubectl port-forward $HEAD_POD 8265:8265

# From separate terminal in same directory as your script
ray job submit --address http://localhost:8265 --runtime-env-json='{"working_dir": ".", "pip": ["wandb"]}' -- python fine_tune_llama_ray.py
# Add other packages to pip list if your script imports them and they are not in the base image.
# If your script and its dependencies are all in the Docker image, runtime_env might not be strictly needed for `working_dir`
# if the script is already in the WORKDIR of the image.