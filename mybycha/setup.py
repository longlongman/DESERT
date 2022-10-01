from setuptools import setup, find_packages

setup(
    name="bycha",
    version="0.1.0",
    keywords=["Natural Language Processing", "Machine Learning"],
    license="MIT Licence",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=open("requirements.txt").readlines(),
    zip_safe=False,

    scripts=[],
    entry_points={
        'console_scripts': [
            'bycha-run = bycha.entries.run:main',
            'bycha-export = bycha.entries.export:main',
            'bycha-preprocess = bycha.entries.preprocess:main',
            'bycha-serve = bycha.entries.serve:main',
            'bycha-serve-model = bycha.entries.serve_model:main',
            'bycha-build-tokenizer = bycha.entries.build_tokenizer:main',
            'bycha-binarize-data = bycha.entries.binarize_data:main'
        ]
    }
)
