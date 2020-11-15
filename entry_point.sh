#!/bin/bash

data_dir="./data/"

echo "Hello from Bash! Unzipping annotated_data.zip"

unzip ./data/annotated_data.zip -d $data_dir
echo "Hello from Bash! Unzipped annotated_data.zip to the: $data_dir"

rm -r ./data/annotated_data.zip
echo "Hello from Bash! Deleted annotated_data.zip"




