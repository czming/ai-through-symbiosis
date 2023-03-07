#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <input_file> <output_file>"
  exit 1
fi

input_file=$1
output_file=$2

i=1
while read line; do
  if [ $((i % 2)) -eq 0 ]; then
    echo $line >> $output_file
  fi
  i=$((i + 1))
done < $input_file
