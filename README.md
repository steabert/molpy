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

After running a Molcas calculation, a module can store wavefunction data
in an HDF5 file. Currently these are only the `scf` and `rasscf` modules
(or `symmetrize` if you have [libmsym](mcodev31/libmsym) support).
The orbital data in the HDF5 file (with extension `.h5`) can be then be examined
or converted like so:

```
# print the orbitals
penny <filename>.h5 --print
# filter to only print coefficients from carbon 2pz basis functions
penny <filename>.h5 --print -m 'C.*2pz'
# convert to Molden format
penny <filename>.h5 --molden
# convert to Gaussian formatted checkpoint format
penny <filename>.h5 --fchk
```

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
With the user-specific installation, the penny program will
end up in `~/.local/bin/penny`, so you either need to add
`~/.local/bin/` to your path, or make a link to the script.

## Help
Questions regarding the use of molpy together with Molcas
can be best asked on the [Molcas Forum](http://www.molcas.org/forum).

Issues with the package itself are handled through the
[issue tracker](https://github.com/steabert/molpy/issues) on github.
