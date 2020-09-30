#!/bin/bash

model_file="./simulatedPlanets/forPaper/training_nsys=3500_seed=2"
data_file="./simulatedPlanets/forPaper/training_nsys=3500.txt"
CVdata_file="./simulatedPlanets/forPaper/test_nsys=1500.txt"

num_planet_features=2
num_stellar_features=2
feature_names="[Rp,P,Teff,logg]"
num_labels=3
batch_size=100
epochs=500
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

results_path="./simulatedPlanets/forPaper/"

#training
if [ $train -eq 1 ]
then
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --plot $plot --saveplot $saveplot --plot_path $plot_path --train "true"
#evaluating performance on holdout test set
elif [ $cross_validate -eq 1 ]
then
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --train "false" --cross_validate "true" --CVdata $CVdata_file
#evaluating performance on training set
else
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --train "false" --cross_validate "false"
fi

