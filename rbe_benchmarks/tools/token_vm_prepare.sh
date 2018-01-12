#!/bin/bash -e

RANDOM_NUMBER=$(cat /dev/urandom | tr -dc 'a-f0-9' | fold -w 8 | head -n 1)
VM_NAME="token-vm-$RANDOM_NUMBER"
FILE_NAME="token-$RANDOM_NUMBER"

echo "New vm name is $VM_NAME."
echo "vm_name=$VM_NAME" > "$_SETUP_OUTPUT"
echo "New vm file name is $FILE_NAME."
echo "file_name=$FILE_NAME" >> "$_SETUP_OUTPUT"
touch "$_SETUP_DONE"
