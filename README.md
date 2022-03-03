# UncertainSCI-tramedemos

This repo houses web app demos for [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI). These demos are reproductions of the examples included with UncertainSCI and are built with [trame](https://kitware.github.io/trame/)

## Running demos

### Setup


To run the web app, the UncertainSCI and trames need to be installed.  It is also best to run trame with Python 3.9 and in a virtual environment.   This can be done by executing the following commands from this repo directory in a terminal window:
```
python3.9 -m venv .venv
source ./.venv/bin/activate
python -m pip install --upgrade pip
pip install -r requiremtents.txt
```

### Running the Web App 

With the environment properly setup, run the web app as follows:
```
python ./1D-python-demos/app.py --port 1234
```
A browser window should open to `http://localhost:1234/`

### Binaries

TODO



