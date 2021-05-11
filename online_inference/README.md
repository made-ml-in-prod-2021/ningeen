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