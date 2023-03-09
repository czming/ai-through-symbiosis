#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 input_file length"
  exit 1
fi

input_file=$1
output_file=$2
length=$(wc -l < $output_file)
count=-1
echocount=0
last_line=$(tail -n 1 $input_file)


while read line; do
  count=$((count + 1))

  if [ $((count % 33)) -eq 0 -a $echocount -lt $length ]; then
    echo $line;
    echocount=$((echocount + 1))
  elif [ $((count % 33)) -eq 7 -a $echocount -lt $length ]; then
    echo $line;
    echocount=$((echocount + 1))
  elif [ $((count % 33)) -eq 14 -a $echocount -lt $length ]; then
    echo $line;
    echocount=$((echocount + 1))
  elif [ $((count % 33)) -eq 21 -a $echocount -lt $length ]; then
    echo $line;
    echocount=$((echocount + 1))
  elif [ $((count % 33)) -eq 27 -a $echocount -lt $length ]; then
    echo $line;
    echocount=$((echocount + 1))
  fi

done < $input_file

if [ $echocount \< $length ]; then
  for (( i=1; i<=length-echocount; i++ )); do
    echo $last_line
  done;
fi


	
