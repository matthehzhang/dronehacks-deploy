from glob import glob
from setuptools import setup

package_name = 'kestrel_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.coursenav', package_name + '.livenav',
              'refnav', 'coursenav', 'onboard', 'onboard.vio'],   # group119, vendored (kestral_nav934am)
    py_modules=['fly_course'],
    # the course file, the published OBJ and the prebuilt voxel map travel with the package
    package_data={package_name: ['data/course/*', 'data/maps/*']},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=False,
    maintainer='Matthew Zhang',
    maintainer_email='vc.matthewzhang@gmail.com',
    description='Team wide-kestrel course planner + localizer + skybeta setpoint mission.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'course_mission = kestrel_nav.course_mission:main',
            'plan_course = kestrel_nav.plan_course:main',
            'localizer = kestrel_nav.coursenav.loc_node:main',
            'live_mission = kestrel_nav.live_mission:main',
            'safe_mission = kestrel_nav.safe_mission:main',
            'skybeta_health = refnav.skybeta_health:main',
            'tune_mission = kestrel_nav.tune_mission:main',
            'host_depth = kestrel_nav.host_depth:main',
        ],
    },
)
