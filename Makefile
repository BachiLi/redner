SHELL := /bin/bash

tf:
	# Compile Tensorflow custom Ops for getting memory addresses of tensors
	# Run this everytime you start a docker container
	bash build_pyrednertensorflow.sh
	
jupyter:
	jupyter notebook --ip=0.0.0.0 --no-browser --allow-root

test-redner:
	python -c 'import redner'