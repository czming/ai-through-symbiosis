die () {
    printf >&2 "$@"
    exit 1
}
[ "$#" -eq 6 ] || die "\n3 arguments required:\n -s split: fraction of data used for training\n -d data: path to folder containing data\n -n number: integer number of folds\n\n $(echo "$#*.5" | bc) provided\n\n"
while getopts s:d:n: flag
do
    case "${flag}" in
        s) split=${OPTARG};;
    	d) dir=${OPTARG};;
	n) n=${OPTARG};;
    esac
done
echo "Training Split: $split"
echo "Data Directory: $dir"

export testing_split="$(bc <<< 1-$split)"
echo "Testing Split: $testing_split"

for fold in $(seq 1 $n)
do 
    echo $fold

    # CREATE SPLITS
      
    # >/dev/null to hide tee stout
    ls $dir | shuf | tee  training-list  testing-list >/dev/null
    export NUM_SAMPLES=$(ls all-labels | wc -l)
    export NUM_TRAINING_SAMPLES=$(printf "%.0f" $(bc <<< $NUM_SAMPLES*$split))
    echo "Training Samples: $NUM_TRAINING_SAMPLES"
    export NUM_TESTING_SAMPLES=$(printf "%.0f" $(bc <<< $NUM_SAMPLES*$testing_split))
    echo "Testing Samples: $NUM_TESTING_SAMPLES"
    
    printf "\nTraining Dataset\n"
    tail -$NUM_TRAINING_SAMPLES training-list > tmp1; mv tmp1 training-list
    cat training-list
    echo ""
    printf "\nTesting Dataset\n"
    head -$NUM_TESTING_SAMPLES testing-list > tmp2; mv tmp2 testing-list
    cat testing-list
    echo ""
    
    # CALCULATE AND APPLY PCA
    
    # mkdir -p pca-training-data
    # rm pca-training-data/*
    # while read filename; do
    #   cp $dir/$filename pca-training-data
    #   cp $dir/$filename data
    #   cp all-labels/$filename.lab label
    # done <training-list

    # python3 PCA.py
    # python3 apply_pca.py

    # mv ./pca-data/* ./data

    while read filename; do
      cp $dir/$filename data
    done <training-list

    # RUN HTK TRAINING SCRIPT

    printf "\nRUN HTK SCRIPT\n"

    cp training-list ./trainsets/training-extfiles0
    ./scripts/prepare_files.sh
    ./scripts/train_parallel.sh ./scripts/options.sh

    # CUSTOM HTK EVALUATINO (WITH PICKLIST SPECIFIC GRAMMAR)
    


    while read filename; do
      HParse grammar/grammar_letter_isolated_ai_general-${filename:9} word.lattice-${filename:9}
      HVite -a -b sil -p 0 -t 0 -s 0 -A -T 1 -H /tmp/models/hmm0.19/newMacros -w /tmp/word.lattice-${filename:9} -S testsets/testing-extfile-${filename:9} -I /tmp/mlf/labels.mlf_tri_internal -i /tmp/ext/result.mlf_letter0 /tmp/dict/dict_letter2letter_ai_general /tmp/commands/commands_letter_isolated_ai_general; mv ext/result.mlf_letter0 results-${filename:9}
      rm word.lattice-${filename:9}
    done <testing-list


    # STORE RESULTS
    mkdir -p fold$fold
    mv results-* fold$fold

    # DELETE TEMPORARY FILES
    rm training-list
    rm testing-list
done
