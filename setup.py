import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ldcast",
    version="0.0.1",
    author="Jussi Leinonen",
    description="Latent diffusion for generative precipitation nowcasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MeteoSwiss/ldcast",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "dask",
        "fire",
        "einops",
        "h5py",
        "matplotlib",
        "netCDF4",
        "numba",
        "numpy",
        "omegaconf",        
        "pyshp",
        "pytorch",
        "pytorch-lightning",
        "scipy",
        "tqdm"
    ],
    extras_require={
        "benchmarks":  ["tensorflow", "pysteps"]
    },
    python_requires='>=3.8',
)
