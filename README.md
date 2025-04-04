# standard-trait-analysis
Data cleanup, exploratory data analysis, and dimensionality reduction of standard phenotypic traits with metadata output from Bloom and sleap-roots-pipeline. 

```
python pipeline/main.py --config "tests/data/base_dev.yaml"
```

```
marimo run pipeline/notebooks/make_csvs.py -- --config_path pipeline_runs/run_2025-04-03_21-09-49/config.yaml
```

```
marimo edit  pipeline/notebooks/make_csvs.py -- --config_path pipeline_runs\run_2025-04-03_21-09-49\config.yaml
```