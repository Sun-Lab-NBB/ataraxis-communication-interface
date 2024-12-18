# Adopt Me
**_Note! Remember to delete this section before releasing v1.0.0 to the public! This section contains 
a set of steps you need to carry out to 'adopt' a project instantiated using this template. Once adopted, this set of
instructions becomes redundant and is safe to remove._**

**Note!** The instructions below assume that you have used this template to instantiate a new repository. 
If you are reading this notice from the base pure-python-template repository, make sure you create your own repository 
from this 
[template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template)
first. 

To adopt this project, follow these steps:
1.  Make sure [miniforge](https://github.com/conda-forge/miniforge) or [miniconda](https://docs.anaconda.com/miniconda/)
    is installed on your development platform. The author of this section encourages using miniforge to benefit from 
    the 'mamba' engine and 'conda-forge' default channel.
2.  Install [tox](https://tox.wiki/en/4.14.2/user_guide.html) into (**_recommended_**) an independent conda 
    environment or, if you mostly work with lab projects, into the base environment (**_this is not recommended_**).
    You can use this command from your conda / mamba shell ```mamba install tox``` or ```conda install tox```.
    Alternatively, you can use ```pip install tox-uv``` to install tox that uses uv instead of pip for maximum runtime 
    speed.
3.  Download this repository to your local machine using your preferred method, such as git-cloning.
4.  Using your shell environment of choice (terminal, zsh, powershell, etc.) ```cd``` into the downloaded project
    directory. Our adoption pipeline works in the current working directory of your shell and verifies key template 
    files are present in the directory before running.
5.  Run ```tox -e adopt``` task and work through the prompts. This task mostly renames the placeholders left through 
    template project files, which effectively converts a generic template into your desired project.
6.  Run ```tox -e import-env```. This will automatically discover the '.yml' file for your os among the files stored in 
    'envs' and install it into your local conda distribution. Generally, this step is not required, but is 
    **_highly recommended_** for running source code outside our automation pipelines.
7.  Verify the information in pyproject.toml file. Specifically, modify the project metadata and add any project-wide 
    dependencies where necessary.
8.  Look through the files inside the 'docs/source/' hierarchy, especially the 'api.rst' and configure it to work for 
    your source code.
9.  Use conda/mamba or pip/uv to add the necessary dependencies to the environment you have imported. Once the 
    environment is configured, use ```tox -e export-env``` to export the environment to the 'envs' folder. The lab 
    requires that each project contains a copy of the fully configured conda environment for each of the operating 
    systems used during development.
10. Remove this section and adjust the rest of the ReadMe to cover the basics of your project. You will tweak it later
    as you develop your source code to better depict your project. Most sections of this readme should already match 
    your project details, so this process is mostly about adding description, usage and maybe additional development 
    requirements.
11. Add source code and tests to verify it works as intended... This is where development truly starts. 
    Run ```tox``` or ```tox --parallel``` before pushing the code back to GitHub to make sure it complies with 
    our standards. See [tox.ini file](tox.ini) for details on available automation pipelines.
12. Congratulations! You have successfully adopted a Sun Lab project. Welcome to the family!
---

# YOUR-PROJECT-NAME

A short (1–2 line max) description of your library (what essential functionality does it provide?)

![PyPI - Version](https://img.shields.io/pypi/v/YOUR-PROJECT-NAME)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/YOUR-PROJECT-NAME)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/YOUR-PROJECT-NAME)
![PyPI - Status](https://img.shields.io/pypi/status/YOUR-PROJECT-NAME)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/YOUR-PROJECT-NAME)
___

## Detailed Description

A long description (1–2 paragraphs max). Should highlight the specific advantages of the library and may mention how it
integrates with the broader Ataraxis project (for Ataraxis-related modules).
___

## Features

- Supports Windows, Linux, and OSx.
- Pure-python API.
- GPL 3 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
___

## Dependencies

For users, all library dependencies are installed automatically for all supported installation methods 
(see [Installation](#installation) section). For developers, see the [Developers](#developers) section for 
information on installing additional development dependencies.
___

## Installation

### Source

1. Download this repository to your local machine using your preferred method, such as git-cloning. Optionally, use one
   of the stable releases that include precompiled binary wheels in addition to source code.
2. ```cd``` to the root directory of the project using your CLI of choice.
3. Run ```python -m pip install .``` to install the project. Alternatively, if using a distribution with precompiled
   binaries, use ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file.

### PIP

Use the following command to install the library using PIP: ```pip install YOUR-PROJECT-NAME```

### Conda / Mamba

**_Note. Due to conda-forge contributing process being more nuanced than pip uploads, conda versions may lag behind
pip and source code distributions._**

Use the following command to install the library using Conda or Mamba: ```conda install YOUR-PROJECT-NAME```
___

## Usage

Add minimal examples on how the end-user can use your library. This section is not to be an in-depth guide on using the
library. Instead, it should provide enough information to start using the library with the expectation that the user
can then study the API documentation and code-hints to figure out how to master the library.

___

## API Documentation

See the [API documentation](https://YOUR-PROJECT-NAME-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library. The documentation also 
covers any available cli/gui-interfaces (such as benchmarks).
___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library. Additionally, it contains instructions for recreating the conda environments
that were used during development from the included .yml files.

### Installing the library

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` to the root directory of the project using your CLI of choice.
3. Install development dependencies. You have multiple options of satisfying this requirement:
    1. **_Preferred Method:_** Use conda or pip to install
       [tox](https://tox.wiki/en/latest/config.html#provision_tox_env) or use an environment that has it installed and
       call ```tox -e import-env``` to automatically import the os-specific development environment included with the
       source code in your local conda distribution. Alternatively, see [environments](#environments) section for other
       environment installation methods.
    2. Run ```python -m pip install .'[dev]'``` command to install development dependencies and the library using 
       pip. On some systems, you may need to use a slightly modified version of this command: 
       ```python -m pip install .[dev]```.
    3. As long as you have an environment with [tox](https://tox.wiki/en/latest/config.html#provision_tox_env) installed
       and do not intend to run any code outside the predefined project automation pipelines, tox will automatically
       install all required dependencies for each task.

**Note:** When using tox automation, having a local version of the library may interfere with tox methods that attempt
to build the library using an isolated environment. It is advised to remove the library from your test environment, or
disconnect from the environment, prior to running any tox tasks. This problem is rarely observed with the latest version
of the automation pipeline, but is worth mentioning.

### Additional Dependencies

In addition to installing the required python packages, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. 
  Currently, this library supports version 3.10 and above. The easiest way to get tox to work as intended is to have 
  separate python distributions, but using [pyenv](https://github.com/pyenv/pyenv) is a good alternative too. 
  This is needed for the 'test' task to work as intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/config.html#provision_tox_env). 
Check [tox.ini file](tox.ini) for details about available pipelines and their implementation.

**Note!** All commits to this library have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime task for this task, use ```tox --parallel```.

### Environments

All environments used during development are exported as .yml files and as spec.txt files to the [envs](envs) folder.
The environment snapshots were taken on each of the three supported OS families: Windows 11, OSx 14.5 and
Ubuntu 22.04 LTS.

To install the development environment for your OS:

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` into the [envs](envs) folder.
3. Use one of the installation methods below:
    1. **_Preferred Method_**: Install [tox](https://tox.wiki/en/latest/config.html#provision_tox_env) or use another
       environment with already installed tox and call ```tox -e import-env```.
    2. **_Alternative Method_**: Run ```conda env create -f ENVNAME.yml``` or ```mamba env create -f ENVNAME.yml```. 
       Replace 'ENVNAME.yml' with the name of the environment you want to install (YOUR_ENV_NAME_osx for OSx, 
       YOUR_ENV_NAME_win for Windows and YOUR_ENV_NAME_lin for Linux).

**Note:** the OSx environment was built against M1 (Apple Silicon) platform and may not work on Intel-based Apple 
devices.

___

## Authors

- YOUR_AUTHOR_NAME.
___

## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
___

## Acknowledgments

- All Sun Lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
