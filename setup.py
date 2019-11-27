# Adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
import os
import re
import sys
import platform
import subprocess
import importlib

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion

class RemoveOldRednerAfterInstall(install):
    def run(self):
        # Remove old redner packages installed by distutils
        from distutils import sysconfig as sc
        site_packages_dir = sc.get_python_lib()
        import shutil
        import glob
        egg_info_path = glob.glob(os.path.join(site_packages_dir, 'redner-0.0.1-*.egg-info'))
        for p in egg_info_path:
            try:
                os.remove(p)
            except:
                print('Warning: detect old redner installation file {} and could not remove it. You may want to remove the file manually.'.format(p))
        lib_path = os.path.join(site_packages_dir, 'redner.so')
        if os.path.exists(lib_path):
            try:
                os.remove(lib_path)
            except:
                print('Warning: detect old redner installation file {} and could not remove it. You may want to remove the file manually.'.format(lib_path))
        data_ptr_lib_path = os.path.join(site_packages_dir, 'libredner_tf_data_ptr.so')
        if os.path.exists(data_ptr_lib_path):
            try:
                os.remove(data_ptr_lib_path)
            except:
                print('Warning: detect old redner installation file {} and could not remove it. You may want to remove the file manually.'.format(data_ptr_lib_path))
        pyredner_path = os.path.join(site_packages_dir, 'pyredner')
        if os.path.exists(pyredner_path):
            try:
                shutil.rmtree(os.path.join(site_packages_dir, 'pyredner'))
            except:
                print('Warning: detect old redner installation file {} and could not remove it. You may want to remove the file manually.'.format(pyredner_path))
        pyredner_tensorflow_path = os.path.join(site_packages_dir, 'pyredner_tensorflow')
        if os.path.exists(pyredner_tensorflow_path):
            try:
                shutil.rmtree(os.path.join(site_packages_dir, 'pyredner_tensorflow'))
            except:
                print('Warning: detect old redner installation file {} and could not remove it. You may want to remove the file manually.'.format(pyredner_tensorflow_path))
        
        install.run(self)

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        super().run()

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                          '-DPYTHON_EXECUTABLE=' + sys.executable]

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if platform.system() == "Windows":
                cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            else:
                cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
                build_args += ['--', '-j8']

            env = os.environ.copy()
            env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                                  self.distribution.get_version())
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        else:
            super().build_extension(ext)

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

# OpenEXR Python installation
openexr_python_version = "1.3.2"
openexr_python_compiler_args = ['-g', '-DVERSION="%s"' % openexr_python_version]
if sys.platform == 'darwin':
    openexr_python_compiler_args.append('-std=c++14')
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(
            get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

setup(name = 'redner',
      version = '0.0.2',
      description = 'A differentiable Monte Carlo ray tracer.',
      author = 'Tzu-Mao Li',
      license = 'MIT',
      packages = packages,
      ext_modules = [CMakeExtension('cmake_example'),
                     Extension('OpenEXR',
                        ['openexrpython/OpenEXR.cpp'],
                        include_dirs=['/usr/include/OpenEXR', '/usr/local/include/OpenEXR', '/opt/local/include/OpenEXR'],
                        library_dirs=['/usr/local/lib', '/opt/local/lib'],
                        libraries=['Iex', 'Half', 'Imath', 'IlmImf', 'z'],
                        extra_compile_args=openexr_python_compiler_args)],
      cmdclass = dict(build_ext=CMakeBuild, install=RemoveOldRednerAfterInstall),
      zip_safe = False)

