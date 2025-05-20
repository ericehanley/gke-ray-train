# Provision a multi-node
gcloud container clusters create eh-inference-gateway-test \
    --zone=europe-west4-a \
    --release-channel=rapid \
    --machine-type=n2-standard-8 \
    --enable-ray-cluster-logging \
    --enable-ray-cluster-monitoring \
    --addons=RayOperator \
    


gcloud container node-pools create gpupool \
    --accelerator type=nvidia-h100-80gb,count=2,gpu-driver-version=latest \
    --project=northam-ce-mlai-tpu \
    --location=europe-west4-a \
    --cluster=eh-inference-gateway-test \
    --machine-type=a3-highgpu-2g \
    --num-nodes=2