from setuptools import setup, find_packages

setup(
    name='GeneticLLM_AutoML',
    version='0.1',
    packages=find_packages(),
    description="Create a pipeline based on LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Israel Campero Jurado and Joaquin Vanschoren",
    author_email="learsi1911@gmail.com",
    url="https://github.com/israel-cj/GeneticLLM_AutoML.git",
    python_requires=">=3.12",
    install_requires=[
        'optuna==4.0.0',
        'openai==1.52.2',
        'pandas==2.2.3',
        'scikit-learn==1.5.2',
        'openml==0.15.0',
        'stopit',  # Add version if known
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    zip_safe=False,
)