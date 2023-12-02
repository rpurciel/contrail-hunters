# syntax=docker/dockerfile:1

FROM continuumio/miniconda3:latest
WORKDIR /usr/src/contrail-hunters
ENV PATH="${PATH}:/usr/src/contrail-hunters"
VOLUME /usr/src/plots

RUN apt-get update --assume-yes
RUN apt-get install libgeos-dev --assume-yes
RUN apt-get install gfortran --assume-yes
COPY environment.yml ./
RUN conda env create -n contrail-hunters -f environment.yml
#RUN conda activate contrail-hunters

COPY . .

CMD ["conda", "run", "--no-capture-output", "-n", "contrail-hunters", "python", "app.py"]
EXPOSE 33507