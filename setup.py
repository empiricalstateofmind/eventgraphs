from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='eventgraphs',
      version='0.1',
      description="""Creating event graphs from temporal network event sequence data (clickstreams, messages, contacts,
                  etc.).""",
      long_description=readme(),
      classifiers=[
        'Programming Language :: Python :: 3.6',
        'Development Status :: v0.1 - Alpha',
        'License :: OSI Approved :: Apache License',
        'Topic :: Data Science :: Temporal Networks',
      ],
      keywords='temporal motifs networks events clickstreams higher-order',
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
          'scipy',
          'ipython'
      ],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )
