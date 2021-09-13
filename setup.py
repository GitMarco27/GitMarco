from distutils.core import setup

setup(
    name='GitMarco',  # How you named your package folder (MyLib)
    packages=['GitMarco'],  # Chose the same as "name"
    version='V0.0.1',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description='An Engineering, Data Science and Deep Learning python package',  # Give a short description about your library
    author='Marco Sanguineti',  # Type in your name
    author_email='marco.sanguineti.info@gmail.com',  # Type in your E-Mail
    url='https://github.com/GitMarco27/GitMarco',  # Provide either the link to your github or to your website
    download_url='https://github.com/GitMarco27/GitMarco/archive/refs/tags/v0.0.1.tar.gz',  # I explain this later on
    keywords=['DeepLearning', 'DataScience', 'Tensorflow'],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        'pysolar',
        'matplotlib',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.8'
    ],
)