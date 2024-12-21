from setuptools import find_packages, setup

package_name = 'ekf_localization'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu2204',
    maintainer_email='ubuntu2204@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf3 = ekf_localization.ekf3:main',
            'ekf_node = ekf_localization.ekf_node:main',
            'ekf_node_task2 = ekf_localization.ekf_node_task2:main',
            'ekf_node_task3 = ekf_localization.ekf_node_task3:main',
            'compute_metrics = ekf_localization.compute_metrics',
            'rosbag2_reader_py = ekf_localization.rosbag2_reader_py',      
        ],
    },
)
