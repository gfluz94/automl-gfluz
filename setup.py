import setuptools

long_description = "This is a library developed to incorporate useful properties and methods in relevant data science packages, such as **scikit-learn** and **pycaret**, in order to provide a pipeline which suits every supervised problem. Therefore, data scientists can spend less time working on building pipelines and use this time more wisely to create new features and tune the best model."

setuptools.setup(
    name="automl-gfluz",
    version="0.0.1",
    author="Gabriel Fernandes Luz",
    author_email="gfluz94@gmail.com",
    description="Package to automate data science and machine learning pipelines.",
    long_description=long_description,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy==1.18.2",
        "pandas==0.25.1",
        "matplotlib==3.2.1",
        "scipy==1.4.1",
        "seaborn==0.9.0",
        "pycaret==1.0.0",
        "scikit-learn==0.22"
    ]
)