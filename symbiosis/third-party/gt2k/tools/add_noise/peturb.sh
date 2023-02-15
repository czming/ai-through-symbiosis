#!/bin/ksh
##################################################################
# All code in the project is provided under the terms specified in
# the file "Public Use.doc" (plaintext version in "Public Use.txt").
#
# If a copy of this license was not provided, please send email to
# haileris@cc.gatech.edu
##################################################################


## input: directory of datafiles, vecotr size
##
## given a directory of datafiles, add gaussian noise to an data vector 
## consisting of all zeros.  The peturbed data will overwrite the original
## files in the specified directory.  a subdirectory will be created and the
## original files will be backed up into this directory.

datafile_dir=$1
vec_size=$2
backup_name=originals
temp_filename=pdat.tmp

BACKUP_DIR=$datafile_dir/$backup_name
PETURB_TMP=$datafile_dir/$temp_filename

## copy the existing datafiles into a backup directory
if [[ ! -a $BACKUP_DIR ]]; then
    mkdir $BACKUP_DIR
fi

## add noise to the exsisting files 
for n in $(ls $datafile_dir ); do
    
    # for all elements in the directory that are not the backup directory
    # or the temporary storage file:
    if [[ $n != $backup_name ]] && [[ $n != $temp_filename ]]; then

	# copy the original file into the backup directory    
	echo "backing up:      $n"
	cp -a $datafile_dir/$n $BACKUP_DIR/$n
	
	# generate the file with noise added to all zero-vectors and
	# save this file to a temprorary file
	echo "adding noise to: $n"
	./gen_noise.sh $datafile_dir/$n $vec_size > $PETURB_TMP

	# copy the contents of the temp file back into the name of the file
	# that was used to generate it
	mv $PETURB_TMP $datafile_dir/$n 
      
    fi

done

