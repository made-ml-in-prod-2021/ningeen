Homework #2
==============================

Run project:
~~~
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
export GMAIL_APP_USERNAME=...
export GMAIL_APP_PASSWORD=...
export GMAIL_USER=...

docker compose up --build
~~~

Airflow variables UI:
~~~
PROD_MODEL=/models/prod/clf.pkl
~~~
---  
Task:
- [X] Поднимите airflow локально, используя docker compose

- [X] Реализуйте dag, который генерирует данные для обучения модели **[5 баллов]**
  
- [X] Реализуйте dag, который обучает модель еженедельно, используя данные за текущий день **[10 баллов]**
  
- [X] Реализуйте dag, который использует модель ежедневно **[5 баллов]**
  
- [X] СРеализуйте сенсоры на то, что данные готовы для дагов тренировки и обучения **[3 балла]**

- [X] все даги реализованы только с помощью DockerOperator **[10 баллов]**
  
- [X] Протестируйте ваши даги **[5 баллов]**

- [ ] В docker compose так же настройте поднятие mlflow и запишите туда параметры обучения, метрики и артефакт(модель)

- [ ] вместо пути в airflow variables используйте апи Mlflow Model Registry
  
- [X] Настройте alert в случае падения дага **[3 балла]**

- [X] проведите самооценку **[1 балл]**

**Summary: 42 балла**