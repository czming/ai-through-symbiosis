array=( 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00 )

for i in "${array[@]}"
do
    python3 it_clustering.py --percent_frames $i
    python3 object_classification_calix.py --percent_frames $i
done
