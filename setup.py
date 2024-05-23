from setuptools import setup

setup(
    name="affine-gaps",
    version="0.1.0",
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    description="Numba-accelerated Python implementation of affine gap penalty extensions for Needleman-Wunsch and Smith-Waterman algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ashvardanian/affine-gaps",
    py_modules=["affine_gaps"],
    install_requires=["numba", "numpy", "colorama"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    extras_require={"dev": ["biopython", "stringzilla", "pytest", "pytest-repeat"]},
    entry_points={
        "console_scripts": [
            "affine-gaps=affine_gaps:main",
        ],
    },
)
