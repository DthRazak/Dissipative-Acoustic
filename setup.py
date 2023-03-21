from setuptools import setup

setup(
    name = 'dissipative-acoustic',
    version = "0.7.0",
    author = "Volodymyr Milchanovskyi",
    author_email = "volodymyr.milchanovskyi@gmail.com",
    packages=["utils", "meshes"],
    package_dir={
        "": ".",
        "utils": "./utils",
        "meshes": "./Master-Thesis/meshes",
    },
)
