from . import export
from . import tools
import copy
import numpy as np


@export
class MolcasMOLDEN:

    def __init__(self, filename, mode):
        self.f = open(filename, mode)
        self.mx_angmom = 3

    def write(self, wfn):
        wfn = copy.deepcopy(wfn)
        wfn.desymmetrize()
        self.write_header()
        self.write_natoms(wfn.natoms_unique)
        self.write_atoms(
            wfn.center_labels,
            wfn.center_charges,
            wfn.center_coordinates,
            )
        self.write_mulliken(
            wfn.mulliken(),
            )
        self.write_gto(
            wfn.basisset,
            )
        self.write_mo(
            tools.scalify(wfn.nbas),
            tools.scalify(wfn.basis_function_ids),
            tools.scalify(wfn.mo_energies),
            tools.scalify(wfn.mo_occupations),
            tools.scalify(wfn.mo_vectors),
            )

    def write_header(self):
        self.f.write('[MOLDEN FORMAT]\n')

    def write_natoms(self, natoms):
        self.f.write('[N_ATOMS]\n')
        self.f.write('{:12d}\n'.format(natoms))

    def write_atoms(self, labels, charges, coords):
        self.f.write('[ATOMS] (AU)\n')
        for idx, (lbl, charge, coord,) in enumerate(zip(labels, charges, coords)):
            self.f.write('{:s} {:7d} {:7d} {:14.7f} {:14.7f} {:14.7f}\n'.format(lbl, idx+1, int(charge), *coord))
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

    def write_mo(self, nbas, ao_labels, mo_energies, mo_occupations, mo_vectors):
        ao_order = tools.argsort(ao_labels, rank=tools.rank_ao_tuple_molden)
        ao_labels = [ao_labels[i] for i in ao_order]

        mo_energies = tools.safe_select(mo_energies, np.zeros(nbas))
        mo_occupations = tools.safe_select(mo_occupations, np.zeros(nbas))

        mo_vectors = mo_vectors[ao_order,:]
        self.f.write('[MO]\n')
        ao_indices = np.array([i for i, (c, n, l, m) in enumerate(ao_labels) if l <= self.mx_angmom])
        for ene, occ, mo in zip(mo_energies, mo_occupations, mo_vectors[ao_indices,:].T):
            self.f.write('Sym = {:s}\n'.format('a1'))
            self.f.write('Ene = {:10.4f}\n'.format(ene))
            self.f.write('Spin = alpha\n')
            self.f.write('Occup = {:10.5f}\n'.format(occ))
            for idx, coef, in enumerate(mo):
                self.f.write('{:4d}   {:16.8f}\n'.format(idx+1, coef))
