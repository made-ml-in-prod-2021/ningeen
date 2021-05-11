Homework #2
==============================

Second homework for "ML in Production"

Docker build...
~~~
docker build -t ningeen/online_inference:v1 .
~~~

...or pull from DockerHub:
~~~
docker pull ningeen/online_inference:v1
~~~

Docker run:
~~~
docker run -p 8000:8000 ningeen/online_inference:v1
~~~

Tests:
~~~
PYTHONPATH=. pytest tests
~~~
---
Оптимизация размера docker image (455Mb):
1. Использование slim-образа (alpine не вышло запустить)
1. Чистка cache при pip install
1. Использование .dockerignore
---
Task:

- [X] ветку назовите homework2, положите код в папку online_inference

- [X] Оберните inference вашей модели в rest сервис(вы можете использовать как FastAPI, так и flask, другие желательно не использовать, дабы не плодить излишнего разнообразия для проверяющих), должен быть endpoint /predict (3 балла)

- [X] Напишите тест для /predict  (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/)

- [X] Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла

- [X] Сделайте валидацию входных данных (например, порядок колонок не совпадает с трейном, типы не те и пр, в рамках вашей фантазии)  (вы можете сохранить вместе с моделью доп информацию, о структуре входных данных, если это нужно) -- 3 доп балла
https://fastapi.tiangolo.com/tutorial/handling-errors/ -- возращайте 400, в случае, если валидация не пройдена

- [X] Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл)

- [X] Оптимизируйте размер docker image (3 доп балла) (опишите в readme.md что вы предприняли для сокращения размера и каких результатов удалось добиться)  -- https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

- [X] опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2 балла)

- [X] напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель (1 балл)
Убедитесь, что вы можете протыкать его скриптом из пункта 3

- [X] проведите самооценку -- 1 доп балл

**Summary: 22 балла**