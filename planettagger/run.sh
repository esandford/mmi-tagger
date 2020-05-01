#!/bin/bash

model_file="./simulatedPlanets/trainingSets/mixedgrammarsoverlapping_mult=zipf_P=logu/mixedgrammarsoverlapping_mult=zipf_P=logu_nsys=10000_seed=2"
data_file="./simulatedPlanets/trainingSets/mixedgrammarsoverlapping_mult=zipf_P=logu/mixedgrammarsoverlapping_mult=zipf_P=logu_nsys=10000.txt"
CVdata_file="./simulatedPlanets/trainingSets/mixedgrammarsoverlapping_mult=zipf_P=logu/mixedgrammarsoverlapping_mult=zipf_P=logu_nsys=4000.txt"

num_planet_features=2
num_stellar_features=3
feature_names="[Rp,P,Teff,logg,Fe/H]"
num_labels=3
batch_size=100
epochs=200
width=2
seed=2
dropout_prob=0.01
lr=0.002

#Bash-interpreted
train=0
cross_validate=1

#Python-interpreted
truth_known="true"
plot="false"
saveplot="false"
plot_path="./simulatedPlanets/crossValidation/testrunscript/plot"

#training
if [ $train -eq 1 ]
then
    python main.py $model_file $data_file --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --plot $plot --saveplot $saveplot --plot_path $plot_path --train "true"
#evaluating performance on holdout test set
elif [ $cross_validate -eq 1 ]
then
    python main.py $model_file $data_file --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --train "false" --cross_validate "true" --CVdata $CVdata_file
#evaluating performance on training set
else
    python main.py $model_file $data_file --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --train "false" --cross_validate "false"
fi

