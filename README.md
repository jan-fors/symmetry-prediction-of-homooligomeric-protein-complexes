# Homomultimer Prediction

## Get Started
```sh
conda env create -f environment.yml
conda activate hmpred
```

## Run Inference


## Run Experiment
1. Create experiment specific config file in `configs/experiments` matching the config requirements from `docs/config.md`
2. Validate the created config with
```sh
python -m src.validate_config <PATH TO CONFIG>
```
3. Run 
```sh
python -m src.run_experiment <PATH TO CONFIG>
```

### Outputs
```sh
result/
├-- best_model.pkl
├-- config.yml
├-- per_class_metrics.png
├-- run.log
├-- test_metrics.json
├-- train_metrics.csv
├-- train_metrics.png
```
