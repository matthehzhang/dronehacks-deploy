from glob import glob
from setuptools import setup

package_name = 'kestrel_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name,'refnav','coursenav','onboard','onboard.vio'],
    py_modules=['fly_course'],
    # the course file, the published OBJ and the prebuilt voxel map travel with the package
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/course', glob('course/*.json')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=False,
    maintainer='Matthew Zhang',
    maintainer_email='vc.matthewzhang@gmail.com',
    description='Depth-only receding-horizon waypoint navigation.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'skybeta_health = refnav.skybeta_health:main',
            'host_depth = kestrel_nav.host_depth:main',
        ],
    },
)
