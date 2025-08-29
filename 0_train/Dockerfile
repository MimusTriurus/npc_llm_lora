FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

COPY main.py /workspace/main.py

ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

CMD ["python3", "main.py"]
