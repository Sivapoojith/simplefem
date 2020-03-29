import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simplefem",
    version="0.0.1",
    author="Nathan",
    author_email="strigusconsilium@gmail.com",
    description="A stupid simple linear elastic FEM solver in python",
    url="https://github.com/heidtn/clive_log",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
