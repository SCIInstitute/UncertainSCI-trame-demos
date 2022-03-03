# UncertainSCI-tramedemos

This repo houses web app demos for [UncertainSCI](https://github.com/SCIInstitute/UncertainSCI). These demos are reproductions of the examples included with UncertainSCI and are built with [trame](https://kitware.github.io/trame/)

## Running demos

### Setup


To run the web app, the UncertainSCI and trames need to be installed.  It is also best to run trame with Python 3.9 and in a virtual environment.   This can be done by executing the following commands from this repo directory in a terminal window:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Web App 

With the environment properly setup, run the web app as follows:


```bash
python ./1D-trame-matplotlib/build_pce_trame.py --port 1234
```

A browser window should open to `http://localhost:1234/`

### Binaries

TODO



