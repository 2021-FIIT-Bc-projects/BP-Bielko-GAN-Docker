FROM jupyter/scipy-notebook

COPY requirements.txt requirements.txt
RUN ["pip", "install", "-r", "requirements.txt"]

COPY x64_dcgan.ipynb x64_dcgan.ipynb
COPY src src

EXPOSE 8888