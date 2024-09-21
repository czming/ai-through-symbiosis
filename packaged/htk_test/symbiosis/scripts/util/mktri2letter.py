#!/usr/bin/env python3

# parses letters back from triletters

fileIn = open('../../commands/commands_tri_internal', 'r')
fileOut = open('../../dict/dict_tri2letter', 'w')

for line in fileIn:
  letter = line[0]
  for i in range(0,len(line)):
    if line[i]=='-':
      letter=line[i+1]
      break
  fileOut.write(letter+' '+line.rstrip()+'\n')

fileIn.close()
fileOut.close()
