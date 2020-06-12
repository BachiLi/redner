#!/bin/bash

rm -R ./dist
rm -R ./build
conda remove -y --name redner-py36 --all
conda remove -y --name redner-py37 --all
conda remove -y --name redner-py38 --all

conda create -y -n redner-py36 python=3.6
conda run -n redner-py36 conda install -y \
        pytorch \
        pybind11 \
        tensorflow \
        -c pytorch
conda run -n redner-py36 pip wheel -w dist --verbose .

conda create -y -n redner-py37 python=3.7
conda run -n redner-py37 conda install -y \
        pytorch \
        pybind11 \
        tensorflow \
        -c pytorch
conda run -n redner-py37 pip wheel -w dist --verbose .

conda create -y -n redner-py38 python=3.8
conda run -n redner-py38 conda install -y \
        pytorch \
        pybind11 \
        -c pytorch
conda run -n redner-py38 pip install tensorflow
conda run -n redner-py38 pip wheel -w dist --verbose .

pip install twine
twine upload dist/redner*.whl
