import setuptools

README = open("README.rst", "r").read()

setuptools.setup(
    name="textory",
    author="Benjamin Roesner",
    author_email="benjamin.roesner@geo.uni-marburg.de",
    description="Image Textures",
    long_description=README,
    url="https://github.com/benr0/textory",
    packages=setuptools.find_packages(),
    #include_package_data  =  True,
    install_requires=["numpy",
                      "xarray",
                      "scipy",
                      "scikit-image",
                      "dask",
                      "distributed",
                      "satpy"],
    setup_requires=["setuptools.scm"],
    use_scm_version=True,
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
