call C:\Users\Administrator\miniconda3\Scripts\activate.bat C:\Users\Administrator\miniconda3

del /s /q dist
del /s /q build
call conda remove -y --name redner-py36 --all
call conda remove -y --name redner-gpu-py36 --all
call conda remove -y --name redner-py37 --all
call conda remove -y --name redner-gpu-py37 --all
call conda remove -y --name redner-py38 --all
call conda remove -y --name redner-gpu-py38 --all

call conda create -y -n redner-py36 python=3.6
call conda run -n redner-py36 conda install -y pytorch pybind11 -c pytorch
call conda run -n redner-py36 conda env config vars set REDNER_CUDA=0
call conda run -n redner-py36 pip wheel -w dist --verbose .

call conda create -y -n redner-gpu-py36 python=3.6
call conda run -n redner-gpu-py36 conda install -y pytorch pybind11 -c pytorch
call conda run -n redner-gpu-py36 conda env config vars set REDNER_CUDA=1
call conda run -n redner-gpu-py36 conda env config vars set PROJECT_NAME=redner-gpu
call conda run -n redner-gpu-py36 pip wheel -w dist --verbose .

call conda create -y -n redner-py37 python=3.7
call conda run -n redner-py37 conda install -y pytorch pybind11 -c pytorch
call conda run -n redner-py37 conda env config vars set REDNER_CUDA=0
call conda run -n redner-py37 pip wheel -w dist --verbose .

call conda create -y -n redner-gpu-py37 python=3.7
call conda run -n redner-gpu-py37 conda install -y pytorch pybind11 -c pytorch
call conda run -n redner-gpu-py37 conda env config vars set REDNER_CUDA=1
call conda run -n redner-gpu-py37 conda env config vars set PROJECT_NAME=redner-gpu
call conda run -n redner-gpu-py37 pip wheel -w dist --verbose .

call conda create -y -n redner-py38 python=3.8
call conda run -n redner-py38 conda install -y pytorch pybind11 -c pytorch
call conda run -n redner-py38 conda env config vars set REDNER_CUDA=0
call conda run -n redner-py38 pip wheel -w dist --verbose .

call conda create -y -n redner-gpu-py38 python=3.8
call conda run -n redner-gpu-py38 conda install -y pytorch pybind11 -c pytorch
call conda run -n redner-gpu-py38 conda env config vars set REDNER_CUDA=1
call conda run -n redner-gpu-py38 conda env config vars set PROJECT_NAME=redner-gpu
call conda run -n redner-gpu-py38 pip wheel -w dist --verbose .

call pip install twine
call twine upload dist/redner*.whl
