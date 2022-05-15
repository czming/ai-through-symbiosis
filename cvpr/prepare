#!/bin/bash
grep "        " HLMTools/Makefile
var1='        '
var2='\t'
sed -i "s&$var1&$var2&g" HLMTools/Makefile
grep "        " HLMTools/Makefile
chmod +x ./configure
linux32 bash
./configure --without-x --disable-hslab
make clean
make all
make install
chmod +x ./HTKDemo/runDemo
chmod +x ./HTKDemo/MakeProtoHMMSet
