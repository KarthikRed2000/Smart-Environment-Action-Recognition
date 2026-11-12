#!/usr/bin/env bash
# Create a GPU VM on GCP, copy the code, and start the experiments in a tmux session.
# Usage (from the repo root, on your laptop):   bash scripts/gcp_setup.sh
# Costs money while the VM exists. Delete it afterwards (see the last line printed).
set -euo pipefail
PROJECT=${PROJECT:-cvproject-514509}
ZONE=${ZONE:-us-central1-a}
VM=${VM:-cs771-stgcn}
MACHINE=${MACHINE:-g2-standard-8}          # 1x NVIDIA L4, 8 vCPU, 32 GB. Alternative: n1-standard-8 + T4
ACCEL=${ACCEL:-}                           # for T4: ACCEL="--accelerator=type=nvidia-tesla-t4,count=1"
IMAGE_FAMILY=${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}
EPOCHS=${EPOCHS:-30}

gcloud config set project "$PROJECT"
gcloud services enable compute.googleapis.com

gcloud compute instances create "$VM" --zone "$ZONE" --machine-type "$MACHINE" $ACCEL \
  --maintenance-policy TERMINATE --provisioning-model SPOT \
  --image-family "$IMAGE_FAMILY" --image-project deeplearning-platform-release \
  --boot-disk-size 100GB --metadata install-nvidia-driver=True

echo "waiting for SSH..."
for i in $(seq 1 60); do      # up to ~15 min (first boot installs the NVIDIA driver)
  gcloud compute ssh "$VM" --zone "$ZONE" --quiet --command "nvidia-smi" >/dev/null 2>&1 && break
  [ "$i" = 60 ] && { echo "VM never became ready -- check it, or delete it: gcloud compute instances delete $VM --zone $ZONE"; exit 1; }
  sleep 15
done
echo "GPU ready"

# copy code only (the VM downloads the dataset itself -- faster than uploading 1.2 GB)
tar czf /tmp/cs771_code.tgz src tests scripts requirements.txt
gcloud compute scp /tmp/cs771_code.tgz "$VM":~ --zone "$ZONE" --quiet
gcloud compute ssh "$VM" --zone "$ZONE" --quiet --command \
  "mkdir -p proj && tar xzf cs771_code.tgz -C proj && cd proj && \
   (python3 -c 'import torch' || python3 -m pip install -q -r requirements.txt) && \
   nohup bash scripts/run_all.sh $EPOCHS 4 > run.log 2>&1 < /dev/null &"

cat <<EOF
Started. Useful commands:
  watch progress : gcloud compute ssh $VM --zone $ZONE --command 'tail -n 2 proj/run.log proj/logs/*.log'
  get results    : gcloud compute scp --recurse $VM:~/proj/results . --zone $ZONE
  DELETE the VM  : gcloud compute instances delete $VM --zone $ZONE
EOF
