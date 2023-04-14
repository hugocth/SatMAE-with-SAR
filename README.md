# SatMAE-with-SAR

This is the repository containing the code used for my paper review.

Before anything else, unzip the SatMAE archive. This folder should contain the folder `SatMAE`.

Then, make sure to download the [datasets models]() by clicking on the embedded link, and place them in the appropriate folders (`fmow-sentinel-2` and `bigearthnet-mm` for the datasets, `SatMAE/trained_models` for the models).
The repo should then look like this:

| --- bigearthnet-mm \
| --- | --- test \
| --- | --- test_10k.csv \
| --- | --- ... \
| --- fmow-sentinel-1 \
| --- | --- ... \
| --- fmow-sentinel-2 \
| --- | --- test \
| --- | --- test_1k.csv \
| --- | --- ... \
| --- SatMAE \
| --- | --- trained_models \
| --- | --- | --- finetune-vit-base-e7.pth \
| --- | --- | --- fintuned-satMAE-MM-e1.pth \
| --- | --- | --- satMAE-MM-e5.pth \
| --- | --- ... \

To run the evaluations :
```
conda env create -f SatMAE/environment.yml
conda activate sat_env
bash SatMAE/fmows2_eval.sh
bash SatMAE/bigearthnet_eval.sh
```
