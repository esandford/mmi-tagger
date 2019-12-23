#!/bin/bash

istart=30
iend=99

main="main.py"

datafile="./realPlanets/allFeatures_3classes/realKOIs_allFeatures.txt"
num_planet_features=2
num_stellar_features=3
feature_names="[Rp,P,Teff,logg,Fe/H]"
num_labels=3
batch_size=50
epochs=250
width=2
dropout_prob=0.01

for i in `seq $istart $iend`; do

    #get a random index between 1 and 50001, inclusive 
    #randIdx=`awk 'BEGIN {
    #   # seed
    #   srand()
    #   print int(1+rand() * 50001)
    #}'`

    #echo $randIdx
    echo $i
    modelfile="./realPlanets/allFeatures_3classes/example-sim-model_realKOIs_allFeatures_seed="$i

    #train
    python main.py $modelfile $datafile --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs  --width $width --seed $i --dropout_prob $dropout_prob --train

    #evaluate
    python main.py $modelfile $datafile --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs  --width $width --seed $i --dropout_prob $dropout_prob 

done