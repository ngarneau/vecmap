#!/bin/bash
#

echo "Pulling the dataset..."
dvc pull

num_runs=10
source_language=en
target_languages=( de es fi it )
for target_language in "${target_languages[@]}"
do
    counter=1
    while [ $counter -le $num_runs ]
    do
        echo "Running experiment for $source_language, $target_language, $counter"
        python3 map_embeddings.py --acl2018 ./data/embeddings/${source_language}.emb.txt ./data/embeddings/${target_language}.emb.txt ./data/embeddings/${source_language}.${target_language}.${counter}.emb.txt ./data/embeddings/${target_language}.${source_language}.${counter}.emb.txt
        python3 eval_translation.py ./data/embeddings/${source_language}.${target_language}.${counter}.emb.txt ./data/embeddings/${target_language}.${source_language}.${counter}.emb.txt ./data/embeddings/${source_language}-${target_language}.test.txt
        ((counter++))
    done
done
