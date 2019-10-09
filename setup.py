from setuptools import setup

setup(
    name='dc2g',
    version='0.1.9',
    keywords='planning, cost-to-go',
    packages=['dc2g'],
    install_requires=[
        'numpy>=1.10.0',
    ],
	# find_links=[
	# 	'/mnt/ubuntu_extra_ssd2/code/dc2g_new/gym-minigrid#egg=gym_minigrid-0.0.5'
	# ]
)
