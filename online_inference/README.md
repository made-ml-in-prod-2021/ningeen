Homework #1
==============================

First homework for "ML in Production"

Create environment:
~~~
conda create -n {environment_name} python=3.8
conda activate {environment_name}
pip install -e .
~~~

Train model:
~~~
python src/train_pipeline.py
~~~
Predict from artifacts:
~~~
python src/predict_pipeline.py artifacts_dir={outputs_folder}
~~~

Test:
~~~
PYTHONPATH=./src/ python -m pytest tests/ 
~~~

Project Organization
------------

    ├── configs                     <- Configs for training and prediction
    │
    ├── data
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── models                      <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                   the creator's initials, and a short `-` delimited description, e.g.
    │                                   `1.0-jqp-initial-data-exploration`.
    │
    ├── reports                     <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── src                         <- Source code for use in this project.
    │   ├── data                    <- code to download or generate data
    │   │
    │   ├── entities                <- params classes
    │   │
    │   ├── features                <- code to turn raw data into features for modeling
    │   │
    │   ├── models                  <- code to train models and then use trained models to make
    │   │
    │   ├── train_pipeline.py       <- code to train model and save artifacts
    │   │
    │   ├── predict_pipeline.py     <- code to predict from artifacts
    │
    ├── requirements.txt            <- The requirements file for reproducing the analysis environment, e.g.
    │                                   generated with `pip freeze > requirements.txt`
    │
    ├── README.md                   <- The top-level README for developers using this project.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
