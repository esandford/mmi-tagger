#!/bin/bash

data_file="./realPlanets_obs/realKOIs_obsFeatures_70percent.txt"
CVdata_file="./realPlanets_obs/realKOIs_obsFeatures_30percent.txt"

num_planet_features=2
num_stellar_features=2
feature_names="[log10(Rp/R*),log10P,Teff,logg]"
batch_size=100
epochs=500
width=2
dropout_prob=0.01
lr=0.002

#Python-interpreted
truth_known="false"
plot="false"
saveplot="false"
plot_path="./simulatedPlanets/crossValidation/testrunscript/plot"

nstart=2
nend=10

sstart=0
send=99

for num_labels in `seq $nstart $nend`; do
  for seed in `seq $sstart $send`; do
    model_file=`echo "./realPlanets_obs/"$num_labels"classes/realKOIs_obsFeatures_"$num_labels"classes_seed="$seed`
    results_path=`echo "./realPlanets_obs/"$num_labels"classes/"`
    echo $model_file
    echo $results_path

    #training
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --plot $plot --saveplot $saveplot --plot_path $plot_path --train "true"
    #evaluating performance on holdout test set
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --train "false" --cross_validate "true" --CVdata $CVdata_file
    #evaluating performance on training set
    python main.py $model_file $data_file --results_path $results_path --num_planet_features $num_planet_features --num_stellar_features $num_stellar_features --feature_names $feature_names --num_labels $num_labels --batch_size $batch_size --epochs $epochs --width $width --seed $seed --dropout_prob $dropout_prob --lr $lr --truth_known $truth_known --train "false" --cross_validate "false"
  done
done



