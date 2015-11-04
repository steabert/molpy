# Welcome to Molpy!
Molpy is a python3 library to make it easy to display Molcas
wavefunction data, compute properties, and convert to other formats.

The library contains classes to read/write Molcas HDF5 (.h5) and INPORB (.???Orb) formats and
write to Gaussian formatted checkpoint (.fchk) and Molden (.molden) format.

The 'penny' program uses the library to manage conversions
between formats and print orbitals in a nicely formatted way.
You can filter AOs (basis functions) by regex match and you
can filter MOs by type or index. This makes it more convenient
to limit viewing of orbitals from MCSCF wavefunctions.

## Installation
Make sure your are using Python 3!
Molpy can be installed for the current user with
```
python setup.py install --user
```
or system-wide with
```
python setup.py install
```

## Help
Questions regarding the use of molpy together with Molcas
can be best asked on the [Molcas Forum](http://www.molcas.org/forum).

Issues with the package itself are handled through the
[issue tracker](https://github.com/steabert/molpy/issues) on github.
