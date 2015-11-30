import copy
import numpy as np

from . import export


@export
class MolcasMOLDEN:
    mx_angmom = 3

    def __init__(self, filename, mode):
        self.f = open(filename, mode)

    def close(self):
        self.f.close()

    def write(self, wfn):
        """
        write wavefunction data to file
        """
        if wfn.basis_set is None:
            raise DataNotAvailable('The fchk format needs basis set info.')

        n_atoms, nuclear_charge = wfn.nuclear_info()
        n_electrons, n_a, n_b, spinmult, electronic_charge = wfn.electronic_info()

        if np.isnan(spinmult):
            spinmult = 1
        if np.isnan(electronic_charge):
            charge = 0
            n_electrons = int(nuclear_charge)
            n_b = (n_electrons - (spinmult - 1)) // 2
            n_a = n_electrons - n_b
        else:
            charge = nuclear_charge + electronic_charge

        self.write_header()
        self.write_natoms(n_atoms)

        basis = wfn.basis_set
        labels = basis.center_labels
        charges = basis.center_charges
        coords = basis.center_coordinates
        self.write_atoms(labels, charges, coords)

        mulliken_charges = wfn.mulliken()
        if np.logical_or.reduce(np.isnan(mulliken_charges)):
            mulliken_charges.fill(0)
        self.write_mulliken(wfn.mulliken())

        self.write_gto(wfn.basis_set.primitive_tree)

        for kind, orbitals in wfn.mo.items():
            orbitals = orbitals.sort_basis(order='molden')
            orbitals = orbitals.limit_basis(limit=self.mx_angmom)
            orbitals.sanitize()
            self.write_mo(orbitals, kind=kind)

    def write_header(self):
        self.f.write('[MOLDEN FORMAT]\n')

    def write_natoms(self, natoms):
        self.f.write('[N_ATOMS]\n')
        self.f.write('{:12d}\n'.format(natoms))

    def write_atoms(self, labels, charges, coords):
        self.f.write('[ATOMS] (AU)\n')
        center_properties = zip(labels, charges, coords)
        template = '{:s} {:7d} {:7d} {:14.7f} {:14.7f} {:14.7f}\n'
        for idx, (lbl, charge, coord,) in enumerate(center_properties):
            self.f.write(template.format(lbl, idx+1, int(charge), *coord))
        self.f.write('[5D]\n')
        self.f.write('[7F]\n')

    def write_mulliken(self, charges):
        self.f.write('[CHARGE] (MULLIKEN)\n')
        for charge in charges:
            self.f.write('{:f}\n'.format(charge))

    def write_gto(self, basisset):
        self.f.write('[GTO] (AU)\n')
        l = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n']
        for center in basisset:
            self.f.write('{:4d}\n'.format(center['id']))
            for angmom in center['angmoms']:
                if angmom['value'] > self.mx_angmom:
                    continue
                for shell in angmom['shells']:
                    self.f.write('   {:1s}{:4d}\n'.format(l[angmom['value']], len(shell['exponents'])))
                    for exp, coef, in zip(shell['exponents'], shell['coefficients']):
                        self.f.write('{:17.9e} {:17.9e}\n'.format(exp, coef))
            self.f.write('\n')

    def write_mo(self, orbitals, kind='restricted'):
        if kind == 'restricted':
            spin = 'alpha'
        else:
            spin = kind
        self.f.write('[MO]\n')
        for irrep, ene, occ, mo in zip(
                orbitals.irreps,
                orbitals.energies,
                orbitals.occupations,
                orbitals.coefficients.T):
            self.f.write('Sym = {:d}\n'.format(irrep))
            self.f.write('Ene = {:10.4f}\n'.format(ene))
            self.f.write('Spin = {:s}\n'.format(spin))
            self.f.write('Occup = {:10.5f}\n'.format(occ))
            for idx, coef, in enumerate(mo):
                self.f.write('{:4d}   {:16.8f}\n'.format(idx+1, coef))
