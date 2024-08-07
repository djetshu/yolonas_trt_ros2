from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'yolo_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all .launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Install all .rviz files from config directory
        (os.path.join('share', package_name, 'config', 'rviz'), glob(os.path.join('config', 'rviz', '*.rviz'))),
    
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jetshu',
    maintainer_email='daffer.queque@outlook.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'yolo_trt = yolo_py.yolo_trt:main',
        ],
    },
)
