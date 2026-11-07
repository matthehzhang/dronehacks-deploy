from glob import glob
from setuptools import setup

package_name = 'cage_test'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/tools', glob('tools/*.py')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='asteyalaxmanan',
    maintainer_email='asteyalaxmanan07@gmail.com',
    description='Cage flight tests: hop_test, tune_test (tuning patterns, logged runs) and frame_check.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'hop_test = cage_test.hop_test:main',
            'tune_test = cage_test.hop_test:tune_main',
            'frame_check = cage_test.frame_check:main',
        ],
    },
)
