# UncertainSCI-tramedemos

This repo houses web app demos for [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI).

These demos are reproductions of the examples included with UncertainSCI and are built with [trame](https://kitware.github.io/trame/)

## Running demos

Demos are organized by directories as indepedent projects.

### Setup


To run the web app, the UncertainSCI and trames need to be installed.  It is also best to run trame with Python 3.9 and in a virtual environment.   This can be done by executing the following commands from this repo directory in a terminal window:

```bash
DEMO_DIR=1D-trame-matplotlib

python3 -m venv venv/${DEMO_DIR}
source venv/${DEMO_DIR}/bin/activate
python -m pip install --upgrade pip
pip install -r ${DEMO_DIR}/requirements.txt
```

### Running the Web App

With the environment properly setup, start the web application as follows:

| Name | Command | URL | Description |
|--|--|--|--|
| `1D-trame-matplotlib` | `python ./1D-trame-matplotlib/build_pce_trame.py --port 1234` | http://localhost:1234/ | [trame][], [matplotlib][] |


A browser window associated with the relevant URL should open automatically.

### Binaries

TODO



