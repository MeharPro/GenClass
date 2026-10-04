#!/bin/bash
# Create jev training nodes one at a time (Mac rule: never parallel az). Usage: scripts/cluster_up.sh NAME:SIZE ...
# Appends "NAME PUBLIC_IP PRIVATE_IP SIZE" to ~/.jev-local/azure_hosts. Backstop auto-shutdown at 18:00 UTC.
set -uo pipefail
RG=rg-jev-train
for spec in "$@"; do
  name="${spec%%:*}"; size="${spec##*:}"
  if grep -q "^$name " ~/.jev-local/azure_hosts 2>/dev/null; then echo "$name exists"; continue; fi
  out=$(az vm create -g $RG -n "vm-jev-$name" --size "$size" --image Canonical:ubuntu-24_04-lts:server:latest \
        --admin-username azureuser --ssh-key-values ~/.ssh/jev_azure.pub --vnet-name vm-jev-trainVNET --subnet vm-jev-trainSubnet \
        --nsg nsg-jev-train --public-ip-sku Standard --accelerated-networking true --ppg ppg-jev --os-disk-size-gb 256 \
        --storage-sku Premium_LRS --os-disk-delete-option Delete --nic-delete-option Delete \
        --tags project=jev-local purpose=train-cluster --query "[publicIpAddress,privateIpAddress]" -o tsv 2>&1)
  if [ $? -ne 0 ]; then echo "FAILED $name $size: $(echo "$out" | grep -i -E 'error|message' | head -2)"; continue; fi
  pub=$(echo "$out" | grep -v -i warning | awk 'NR==1'); priv=$(echo "$out" | grep -v -i warning | awk 'NR==2')
  echo "$name $pub $priv $size" >> ~/.jev-local/azure_hosts
  az vm auto-shutdown -g $RG -n "vm-jev-$name" --time 1800 -o none
  echo "created $name $size $pub $priv"
done
