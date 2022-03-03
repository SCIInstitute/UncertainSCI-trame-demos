# UncertainSCI Web Demos

This project is a collection of web applications illustrating how to use [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI).

They are based on the examples included with UncertainSCI.
See https://uncertainsci.readthedocs.io/en/latest/tutorials/index.html

They are built with [trame][].

## Running demos

Demos are organized by directories as indepedent projects.

### Setup

To run each web app, specific requirements need to be installed.

It is also best to run the application with Python >= 3.7 and in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).

This can be done by executing the following commands in a terminal window:

```bash
# Download the source
git clone git@github.com:SCIInstitute/UncertainSCI-web-demos.git

# ... and change the current working directory
cd UncertainSCI-web-demos

# Choose the demo
DEMO_DIR=1D-trame-matplotlib

# Create virtual environment
python3 -m venv venv/${DEMO_DIR}

# Activate the virtual env
source venv/${DEMO_DIR}/bin/activate

# Update the pip version associated with the virtual env
python -m pip install --upgrade pip

# Install demo requirements using pip
pip install -r ${DEMO_DIR}/requirements.txt
```

### Running the Web App

With the environment properly setup, start the web application as follows:

| Name | Command | Frameworks |
|--|--|--|
| `1D-trame-matplotlib` | `python ./1D-trame-matplotlib/build_pce_trame.py --port 1234` | [trame][], [matplotlib][] |
| `1D-trame-plotly` | `python ./1D-trame-plotly/app.py --port 1235` | [trame][], [plotly][] |

A browser window associated with the relevant URL of the form `http://localhost:<port>` should open automatically.

### Binaries

TODO


[trame]: https://kitware.github.io/trame/
[matplotlib]: https://matplotlib.org/
[plotly]: https://plotly.com/

