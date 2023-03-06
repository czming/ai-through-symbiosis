#!/bin/bash

# Get filename from command line argument
filename=$1

# Get number of characters in file
char_count=$(wc -c < "$filename")

# get picklist_num

numbers=$(echo "$filename" | awk -F"_" '{print $2, $4}')



# Subtract 1 from char count and divide by 2 using bc
half_count=$(echo "($char_count - 1) / 2" | bc)

output="\$char = a e i m;\n"
output+="(sil "
for i in $(seq 1 ${half_count}); do
    output+="\$char "
done
output+="sil)"
echo -e "$output" > grammar_letter_isolated_ai_general-${numbers}





