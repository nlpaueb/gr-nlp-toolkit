import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gr-nlp-toolkit",
    version="0.0.2",
    author="nlpaueb",
    author_email="p3170148@aueb.gr, p3170039@aueb.gr",
    description="A Transformer-based Natural Language Processing Pipeline for Greek",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nlpaueb/gr-nlp-toolkit",
    project_urls={
        #"Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Greek"
    ],
    package_dir={"": "gr_nlp_toolkit"},
    packages=setuptools.find_packages(where="gr_nlp_toolkit"),
    python_requires=">=3.6",
)