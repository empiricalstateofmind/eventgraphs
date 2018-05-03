from setuptools import setup

setup(name='eventgraphs',
      version='0.1',
      description='Finding temporal components and motifs in temporal networks',
      long_description='Finding temporal components and motifs in temporal networks',
      classifiers=[
        'Programming Language :: Python :: 3.5',
      ],
      keywords='temporal motifs networks',
      url='https://github.com/empiricalstateofmind/eventgraph',
      author='Andrew Mellor',
      author_email='mellor91@hotmail.co.uk',
      license='Apache License, Version 2.0',
      packages=['eventgraphs'],
      install_requires=[
          'networkx',
          'pandas',
          'numpy',
          'matplotlib',
          'scipy'
      ],
      include_package_data=True,
      zip_safe=False)
