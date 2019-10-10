example run command:

python main.py ./simulatedPlanets/oneGrammar_distinctRp/example-sim-model_allFeatures_uniformP ./simulatedPlanets/oneGrammar_distinctRp/fake_grammaticalSystems_allFeatures_uniformP.txt --num_planet_features 2 --num_stellar_features 3 --feature_names [Rp,P,Teff,logg,FeH] --num_labels 3 --batch_size 100 --epochs 100  --width 2 --seed 1 --dropout_prob 0.01 --train --plot