# standard-trait-analysis
Data cleanup, exploratory data analysis, and dimensionality reduction of standard phenotypic traits with metadata output from Bloom and sleap-roots-pipeline. 

```
python pipeline/main.py --config tests/data/base_dev.yaml --edit --export-html
python pipeline/main.py --config tests/data/base_dev.yaml --export-html
```

```
marimo run pipeline/notebooks/make_csvs.py -- --config_path pipeline_runs\run_2025-04-04_12-38-15\config.yaml
```

```
marimo edit  pipeline/notebooks/make_csvs.py -- --config_path pipeline_runs\run_2025-04-04_12-38-15\config.yaml
```

```
marimo edit  pipeline/notebooks/data_cleanup.py -- --config_path pipeline_runs\run_2025-04-04_12-38-15\config.yaml
```

```
python pipeline/main.py --config 20250516_config_wheat_cleanup.yaml --edit
marimo edit  pipeline/notebooks/data_cleanup.py -- --config_path pipeline_runs/run_2025-05-17_11-59-11/config.yaml
```

```
python pipeline/main.py --config 20250517_config_wheat_cleanup.yaml --edit
marimo edit  pipeline/notebooks/data_cleanup.py -- --config_path pipeline_runs/run_2025-05-17_17-14-41/config.yaml
```

```
python pipeline/main.py --config 20250518_config_wheat_cleanup.yaml --edit
```

```
python pipeline/main.py --config 20250520_config_wheat_cleanup.yaml --edit
```