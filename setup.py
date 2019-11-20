import setuptools

setuptools.setup(
    name = "textory",
    version = "0.1.0b",
    author = "Benjamin Roesner",
    author_email = "benjamin.roesner@geo.uni-marburg.de",
    description = "Image Textures",
    long_description = "",
    long_description_content_type = "text/markdown",
    url = "https://github.com/benr0/textory",
    packages = setuptools.find_packages(),
    #include_package_data  =  True,
    install_requires=["numpy",
                      "xarray",
                      "scipy",
                      "scikit-image",
                      "dask",
                      "distributed",
                      "satpy"],
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
