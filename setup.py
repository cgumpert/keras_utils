import setuptools

setuptools.setup(
    name="keras_utils",
    version="0.0.1",
    author="Christian Gumpert",
    description="Collection of helpful utility functions for working with keras",
    url="https://github.com/cgumpert/keras_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["keras", "logging", "numpy>=1.15.4,<1.16", "pandas", "tensorflow"]
)
