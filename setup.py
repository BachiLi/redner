from distutils.core import setup
import importlib

torch_spec = importlib.util.find_spec("torch")
tf_spec = importlib.util.find_spec("tensorflow")
packages = []
if torch_spec is not None:
    packages.append('pyredner')
if tf_spec is not None:
    packages.append('pyredner_tensorflow')
if len(packages) == 0:
    print('Error: PyTorch or Tensorflow must be installed.')
    exit()

setup(name = 'redner',
      version = '0.0.1',
      description = 'Redner',
      packages = packages)
