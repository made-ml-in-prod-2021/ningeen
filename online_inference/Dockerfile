FROM python:3.8-slim
ENV PYTHONPATH "${PYTHONPATH}:."

COPY . online_inference/
RUN pip install --no-cache-dir -r online_inference/requirements.txt
WORKDIR online_inference

CMD ["python", "src/app.py"]