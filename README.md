# classifier-demo

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

> Playground with machine learning methods using python sklearn. 

This is more like a playground project that provides basic examples to evaluate and test classifiers using a popular dataset. We explore machine learning methods using the python sklearn package.

## Table of Contents

- [Install](#install)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)

## Install
- Clone the repository and enter the project directory.

- Create your virtual environment and install the required dependencies:

```
virtualenv -p `which python3` venv
source venv/bin/activate
pip install
```

- Make sure you have a python framework like Anaconda. Then use `pythonw` to run the examples.

## Usage
In the `src` folder:

Test the versions of the installed libraries:

```
pythonw check_versions.py
```

Inspect the dataset:

```
pythonw inspect-dataset.py
```

Train some algorithms and evaluate them:

```
pythonw evaluate-algos.py
```

Browse the `src` folder for more examples.

## Support
If you're having any problem, please raise an issue on GitHub.

## Contributing
PRs accepted. Some general guidelines:

- Write a concise commit message explaining your changes.
- If applies, write more descriptive information in the commit body.
- Refer to the issue/s your pull request fixes (if there are issues in the github repo).
- Write a descriptive pull request title.
- Squash commits when possible.

Before your pull request can be merged, the following conditions must hold:

- All the tests passes (if any).
- The coding style aligns with the project's convention.
- Your changes are confirmed to be working.



Small note: If editing the Readme, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme) specification.

## License
The project is licensed under the Apache 2.0 license.
