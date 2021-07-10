This project aims at providing flexiable BCI online or offline experiment design framework.

Supported Language
- Python

## Installation:

**Must by running Python 3.6**

To install, fork or clone the repository and go to the downloaded directory,
then run

```
pip install -r requirements.txt
python setup.py develop    # because no stable release yet
```

### Requirements we use
- Microsoft Visual Studio with C++ compiler
- numpy
- scipy
- psychopy == 3.0.0

### Notes
1. you may need inpout32.dll inpoutx64.dll driver to use parallel port in psychopy, we provide inpout32.dll driver in examples. You should put this driver under you system search path or just with your experiment python file together. For inpoutx64.dll, a tricky way is just to rename inpout32.dll to inpoutx64.dll.