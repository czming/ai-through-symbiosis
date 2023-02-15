#!/bin/bash

cnt=71
while read LINE
do
    echo $LINE > "picklist_${cnt}_raw.txt"
    cnt=$((cnt+1))
done < clipboard.txt
