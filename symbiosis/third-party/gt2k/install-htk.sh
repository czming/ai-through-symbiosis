#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################

##################################################################
# install-htk.sh
# 
# Simple installer for HTK for Linux systems.  Sets up and compiles
# HTK in a local directory, then copies the binaries to their final
# destination
##################################################################
HTK_SRC=$1;
HTK_DEST=$2;
echo "Installing HTK from $1 to $2."
echo "BE SURE THAT YOU HAVE WRITE PERMISSIONS FOR $2"

if [ ! -e ${HTK_SRC} ]; then
	echo "Couldn't read ${HTK_SRC}!!  Exiting..."
	exit 1;
fi

# unpack HTK and move into the directory
echo "Unpacking ${HTK_SRC}..."
tar zxvf ${HTK_SRC}
cd htk/

# set up the environment
echo "Setting up environment and compilation settings..."
. env/exp.linux

# install locally, move later
export HBIN=`pwd`
mkdir ./bin.linux

#
echo "Compiling in HTKLib/"
cd HTKLib/
make 

echo "Compiling in HTKTools/"
cd ../HTKTools/
make

echo "Compiling in HLMLib/"
cd ../HLMLib/
make

echo "Compiling in HLMTools/"
cd ../HLMTools/
make

# Move the resulting stuff
cd ..
echo "Installing files in ${HTK_DEST}..."
for i in ./bin.linux/*
do
	echo "$i -> ${HTK_DEST}/${i#./bin.linux/}"
	install -D -b $i ${HTK_DEST}/${i#./bin.linux/}
done
