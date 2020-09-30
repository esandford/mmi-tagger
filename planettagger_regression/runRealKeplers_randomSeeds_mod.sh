#!/bin/bash

data_file="./realPlanets_mod/realKOIs_modFeatures_70percent.txt"
CVdata_file="./realPlanets_mod/realKOIs_modFeatures_30percent.txt"

num_planet_features=3
num_stellar_features=2
feature_names="[log10Rp,log10a,log10Finsol,log10R*,log10M*]"
batch_size=100
epochs=500
width=2
dropout_prob=0.01
lr=0.02

#Python-interpreted
truth_known="false"
plot="false"
saveplot="false"
plot_path="./simulatedPlanets/crossValidation/testrunscript/plot"

sstart=0
send=99

num_labels=0

for seed in `seq $sstart $send`; do
    model_file=`echo "./realPlanets_mod/realKOIs_modFeatures_seed="$seed`
    results_path=`echo "./realPlanets_mod/"`
    echo $model_file
    echo $results_path

    #training
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --num_labels $num_labels --feature_names $feature_names --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --plot $plot --saveplot $saveplot --plot_path $plot_path --train "true"
    #evaluating performance on holdout test set
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --num_labels $num_labels --feature_names $feature_names --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --train "false" --cross_validate "true" --CVdata $CVdata_file
    #evaluating performance on training set
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --num_labels $num_labels --feature_names $feature_names --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --train "false" --cross_validate "false"
done



