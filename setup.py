from setuptools import find_packages, setup

package_name = 'pf_localization'

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
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pf_node = pf_localization.pf_node:main',
            'compute_metrics = pf_localization.compute_metrics',
            'pf_node_task2 = pf_localization.pf_node_task2:main',
            'plot_landmark = pf_localization.plot_landmark:main',
        ],
    },
)
