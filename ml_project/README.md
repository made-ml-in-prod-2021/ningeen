ml_project
==============================

First homework for "ML in Production"

Installation: 
~~~
*TODO*
~~~
Train model:
~~~
python src/train_pipeline.py configs/{config_name}.yaml 
~~~
Predict from artifacts:
~~~
python src/predict_pipeline.py data_path transformer_path model_path output_path
~~~

Test:
~~~
PYTHONPATH=./src/ python -m pytest tests/ 
~~~

Project Organization
------------
*TODO*