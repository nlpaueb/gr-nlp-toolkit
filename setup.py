import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gr-nlp-toolkit",
    version="0.2.0",
    author="nlpaueb",
    author_email="p3170148@aueb.gr, p3170039@aueb.gr, spirosbarbakos@gmail.com,  eleftheriosloukas@aueb.gr, ipavlopoulos@aueb.gr",
    description="The state-of-the-art NLP toolkit for (modern) Greek",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nlpaueb/gr-nlp-toolkit",
    project_urls={
        # "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Greek",
    ],
    packages=setuptools.find_packages(where=".", exclude="./tests"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.2",
        "transformers>=4.11.1",
        "huggingface_hub>=0.23.5",
    ],
)
