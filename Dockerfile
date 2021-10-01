FROM jupyter/scipy-notebook
ENV GRANT_SUDO=yes

COPY requirements.txt requirements.txt

#dependencies
RUN ["pip", "install", "-r", "requirements.txt"]

EXPOSE 8888