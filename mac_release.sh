#!/bin/bash

rm -R ./dist
conda remove -y --name redner-py36 --all
conda remove -y --name redner-py37 --all

conda create -y -n redner-py36 python=3.6
conda run -n redner-py36 conda install -y \
        pytorch \
        pybind11 \
        tensorflow=1.14.0 \
        -c pytorch
conda run -n redner-py36 pip wheel -w dist --verbose .

conda create -y -n redner-py37 python=3.7
conda run -n redner-py37 conda install -y \
        pytorch \
        pybind11 \
        tensorflow=1.14.0 \
        -c pytorch
conda run -n redner-py37 pip wheel -w dist --verbose .

pip install twine
twine upload dist/redner*.whl