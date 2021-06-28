Homework #4
==============================

Install minicube:
~~~
https://minikube.sigs.k8s.io/docs/start/
~~~

Deploy pods:
~~~
kubectl apply -f kubernetes_manifests/online-inference-pod.yaml
~~~

---  
Task:
- [X] Установите kubectl

- [X] Разверните kubernetes (minicube). Убедитесь, с кластер поднялся **[5 баллов]**

- [X] Напишите простой pod manifests. Задеплойте приложение в кластер **[4 балла]**

- [X] Пропишите requests/limits **[2 балла]**

- [X] Модифицируйте свое приложение так, чтобы оно стартовало не сразу(с задержкой секунд 20-30) и падало спустя минуты работы. Добавьте liveness и readiness пробы **[3 балла]**

- [X] Создайте replicaset, сделайте 3 реплики вашего приложения **[3 балла]**

- [X] Опишите деплоймент для вашего приложения **[3 балла]**

**Summary: 20 баллов**