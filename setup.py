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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research'
      ],
      keywords='temporal motifs networks events clickstreams higher-order',
      project_urls={
        'Source': 'https://github.com/empiricalstateofmind/eventgraph',
        'Tracker': 'https://github.com/empiricalstateofmind/eventgraph/issues',
      },
      author='Andrew Mellor',
      author_email='mellor91@hotmail.co.uk',
      license='Apache Software License',
      packages=['eventgraphs'],
      python_requires='>=3',
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
