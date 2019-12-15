#!/bin/sh

rm -R ./build
sudo rm -Rf ./dist
sudo docker rmi tzumao/redner:manylinux tzumao/redner:manylinux-gpu
mkdir -p dist
sudo docker build -t tzumao/redner:manylinux -f manylinux.Dockerfile .
sudo docker create --name redner_manylinux tzumao/redner:manylinux --entrypoint /
sudo docker cp redner_manylinux:/dist .
sudo docker build -t tzumao/redner:manylinux-gpu -f manylinux-gpu.Dockerfile .
sudo docker create --name redner_manylinux_gpu tzumao/redner:manylinux-gpu --entrypoint /
sudo docker cp redner_manylinux_gpu:/dist .
sudo docker rm -f redner_manylinux
sudo docker rm -f redner_manylinux_gpu

python -m pip install --upgrade twine
python -m twine upload ./dist/redner*manylinux*.whl

