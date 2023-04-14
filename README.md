# SatMAE-with-SAR

This is the repository containing the code used for my paper review.

Before anything else, unzip the SatMAE archive. This folder should contain the folder `SatMAE`.

Then, make sure to download the [datasets]() and [models]() by clicking on the links (coming soon), and place them in the appropriate folders (`fmow-sentinel-2` and `bigearthnet-mm` for the datasets, `SatMAE/trained_models` for the models).

To run the evaluations :
```
conda env create -f SatMAE/environment.yml
conda activate sat_env
bash SatMAE/fmows2_eval.sh
bash SatMAE/bigearthnet_eval.sh
```
