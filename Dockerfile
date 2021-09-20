FROM jupyter/scipy-notebook
COPY requirements.txt requirements.txt

#dependencies
RUN ["pip", "install", "-r", "requirements.txt"]

EXPOSE 8888