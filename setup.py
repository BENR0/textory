import setuptools

setuptools.setup(
    name = "textory",
    version = "0.0.1b",
    author = "Benjamin Roesner",
    author_email = "benjamin.roesner@geo.uni-marburg.de",
    description = "Image Textures",
    long_description = "",
    long_description_content_type = "text/markdown",
    url = "https://github.com/benr0/textury",
    packages = setuptools.find_packages(),
    #include_package_data  =  True,
    install_requires=["numpy",
                      "xarray",
                      "scipy",
                      "scikit-image",
                      "dask",
                      "distributed",
                      "satpy"],
    #package_data={
        #"hyfog": ["data/rasterfiles/*",
                  #"data/shapefiles/*",
                  #"data/station_metadata/*"
                  #"opencl/opencl/*"],
    #},
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
