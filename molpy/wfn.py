import numpy as np
from collections import namedtuple
from scipy import linalg as la
import re

from . import export
from .basis import BasisSet
from .orbitals import OrbitalSet
from .mh5 import MolcasHDF5
from .inporb import MolcasINPORB
from .errors import Error, DataNotAvailable


@export
class Wavefunction():
    def __init__(self, mo, basis_set, salcs=None,
                 overlap=None, fockint=None,
                 spinmult=None, n_bas=None, n_sym=None):
        self.mo = mo
        if 'alpha' in mo and 'beta' in mo:
            self.unrestricted = True
        elif 'restricted' in mo:
            self.unrestricted = False
        else:
            raise Exception('invalid key(s) in mo dict')
        self.basis_set = basis_set
        self.salcs = salcs
        self.overlap = overlap
        self.fockint = fockint
        self.spinmult = spinmult
        self.n_bas = n_bas
        self.n_sym = n_sym

    def electronic_info(self):
        '''
        return a tuple containing the total number of electrons, the number of
        alpha electrons, the number of beta electrons, and the spin multiplicity.
        When occupation numbers are not available (i.e. NaNs), the number of
        electrons will be set to represent a neutral system.
        '''
        if self.unrestricted:
            n_alpha = int(np.sum(self.mo['alpha'].occupations))
            n_beta = int(np.sum(self.mo['beta'].occupations))
            n_electrons = n_alpha + n_beta
            if self.spinmult is None:
                spinmult = n_alpha - n_beta + 1
            else:
                spinmult = self.spinmult
        else:
            n_electrons = np.sum(self.mo['restricted'].occupations)
            if self.spinmult is None:
                spinmult = 1
            else:
                spinmult = self.spinmult
            n_beta = (n_electrons - (spinmult - 1)) // 2
            n_alpha = n_electrons - n_beta
        electronic_charge = -n_electrons
        return (n_electrons, n_alpha, n_beta, spinmult, electronic_charge)

    def nuclear_info(self):
        '''
        return a tuple containing the total number of atoms and nuclear charge
        '''
        n_atoms = len(self.basis_set.center_labels)
        nuclear_charge = int(np.sum(self.basis_set.center_charges))
        return (n_atoms, nuclear_charge)

    def print_orbitals(self, types=None, erange=None, pattern=None, order=None):

        for kind in ('restricted', 'alpha', 'beta'):
            if kind not in self.mo:
                continue
            else:
                orbitals = self.mo[kind]

            if types is not None:
                orbitals = orbitals.type(*types)

            if erange is not None:
                orbitals = orbitals.erange(*erange)

            if pattern is not None:
                orbitals = orbitals.pattern(pattern)

            if order is not None:
                orbitals = orbitals.sort_basis(order=order)

            self._print_mo_header(kind=kind)
            if self.n_sym > 1:
                orbitals.show_by_irrep()
            else:
                orbitals.show()

    def guessorb(self):
        """
        return a set of molecular orbitals that diagonalize the atomic fock matrix.
        """
        if self.overlap is None or self.fockint is None:
            raise DataNotAvailable('guessorb is missing the overlap and/or fockint matrix')

        guessorb = {}
        Smat_ao = np.asmatrix(self.overlap)
        Fmat_ao = np.asmatrix(self.fockint)
        Fmat_ao = Smat_ao.T * Fmat_ao * Smat_ao
        for kind in self.mo.keys():
            C_mo = np.asmatrix(self.mo[kind].coefficients)
            E_mo = np.empty(len(self.mo[kind].energies))
            irreps = self.mo[kind].irreps.copy()
            for irrep in np.unique(irreps):
                mo_set, = np.where(irreps == irrep)
                Cmat = C_mo[:,mo_set]
                Smat_mo = Cmat.T * Smat_ao * Cmat
                # orthonormalize
                s,U = np.linalg.eigh(Smat_mo)
                U_lowdin = U * np.diag(1/np.sqrt(s)) * U.T
                Cmat = Cmat * U_lowdin
                # diagonalize metric Fock
                Fmat_mo = Cmat.T * Fmat_ao * Cmat
                f,U = np.linalg.eigh(Fmat_mo)
                Cmat = Cmat * U
                # copy back to correct supsym id
                C_mo[:,mo_set] = Cmat
                E_mo[mo_set] = f
            # finally, create new orbital set with new coefficients and energies
            mo_order = np.argsort(E_mo)
            guessorb[kind] = OrbitalSet(C_mo[:,mo_order],
                                        energies=E_mo[mo_order],
                                        irreps=irreps[mo_order],
                                        basis_set=self.mo[kind].basis_set)
        return guessorb

    def mulliken(self):
        """
        perform a mulliken population analysis
        """
        if self.overlap is not None:
            Smat_ao = np.asmatrix(self.overlap)
        else:
            raise Exception('mulliken analysis is missing the overlap matrix')

        population = {}
        for kind, mo in self.mo.items():
            population[kind] = np.zeros(len(self.basis_set.center_charges))
            Cmat = np.asmatrix(mo.coefficients)
            D = Cmat * np.diag(mo.occupations) * Cmat.T
            DS = np.multiply(D, Smat_ao)
            for i, (ao, basis_id) in enumerate(zip(np.asarray(DS), mo.basis_ids)):
                pop = np.sum(ao)
                cgto_tuple = mo.basis_set.contracted_ids[basis_id]
                center_id, l, n, m = cgto_tuple
                population[kind][center_id-1] += pop

        if self.unrestricted:
            population_total = population['alpha'] + population['beta']
        else:
            population_total = population['restricted']
        mulliken_charges = self.basis_set.center_charges - population_total
        return mulliken_charges

    @staticmethod
    def _print_mo_header(kind=None, width=128, delim='*'):
        if kind is not None:
            text = kind + ' molecular orbitals'
        else:
            text = 'molecular orbitals'
        starline = delim * width
        starskip = delim + ' ' * (width - 2) + delim
        titleline = delim + text.title().center(width - 2, ' ') + delim
        print('\n'.join([starline, starskip, titleline, starskip, starline, '']))

    @classmethod
    def from_h5(cls, filename):
        """ Generates a wavefunction from a Molcas HDF5 file """
        f = MolcasHDF5(filename, 'r')

        n_bas = f.n_bas

        if n_bas is None:
            raise Exception('no basis set size available on file')

        n_sym = len(n_bas)

        if n_sym > 1:
            n_atoms = f.natoms_all()
            center_labels = f.desym_center_labels()
            center_charges = f.desym_center_charges()
            center_coordinates = f.desym_center_coordinates()
            contracted_ids = f.desym_basis_function_ids()
        else:
            n_atoms = f.natoms_unique()
            center_labels = f.center_labels()
            center_charges = f.center_charges()
            center_coordinates = f.center_coordinates()
            contracted_ids = f.basis_function_ids()
        primitive_ids = f.primitive_ids()
        primitives = f.primitives()


        basis_set = BasisSet(
                center_labels,
                center_charges,
                center_coordinates,
                contracted_ids,
                primitive_ids,
                primitives,
                )

        if n_sym > 1:
            mo_irreps = np.empty(sum(n_bas), dtype=np.int)
            offset = 0
            for irrep, nb in enumerate(n_bas):
                mo_irreps[offset:offset+nb] = irrep
                offset += nb
            salcs = f.desym_matrix()
        else:
            mo_irreps = f.supsym_irrep_indices()

        unrestricted = f.unrestricted()
        if unrestricted:
            kinds = ['alpha', 'beta']
        else:
            kinds = ['restricted']

        mo = {}
        for kind in kinds:
            mo_occupations = f.mo_occupations(kind=kind)
            mo_energies = f.mo_energies(kind=kind)
            mo_typeindices = f.mo_typeindices(kind=kind)
            mo_vectors = f.mo_vectors(kind=kind)
            mo_vectors = cls.reshape_square(mo_vectors, n_bas)

            if n_sym > 1:
                mo_vectors = np.dot(salcs, mo_vectors)

            mo[kind] = OrbitalSet(mo_vectors,
                                  types=mo_typeindices,
                                  irreps=mo_irreps,
                                  energies=mo_energies,
                                  occupations=mo_occupations,
                                  basis_set=basis_set)

        try:
            overlap = f.ao_overlap_matrix()
            overlap = cls.reshape_square(overlap, n_bas)
            if overlap is not None and n_sym > 1:
                overlap = np.dot(np.dot(salcs, overlap), salcs.T)
        except DataNotAvailable:
            overlap = None

        try:
            fockint = f.ao_fockint_matrix()
            fockint = cls.reshape_square(fockint, n_bas)
            if fockint is not None and n_sym > 1:
                fockint = np.dot(np.dot(salcs, fockint), salcs.T)
        except DataNotAvailable:
            fockint = None

        try:
            ispin = f.ispin()
        except DataNotAvailable:
            ispin = None

        return cls(mo, basis_set, salcs=salcs,
                   overlap=overlap, fockint=fockint,
                   spinmult=ispin, n_sym=n_sym, n_bas=n_bas)

    @classmethod
    def from_inporb(cls, filename):
        """ Generates a wavefunction from a Molcas INPORB file """
        f = MolcasINPORB(filename, 'r')

        n_bas = f.n_bas

        if n_bas is None:
            raise Exception('no basis set size available on file')

        n_sym = len(n_bas)

        irrep_list = [[irrep]*nb for irrep, nb in enumerate(n_bas)]
        mo_irreps = np.concatenate(irrep_list)

        unrestricted = f.unrestricted
        if unrestricted:
            kinds = ['alpha', 'beta']
        else:
            kinds = ['restricted']

        mo = {}
        for kind in kinds:
            f.rewind()
            # order of reading matters!
            mo_vectors = f.read_orb(kind=kind)
            mo_occupations = f.read_occ(kind=kind)
            mo_energies = f.read_one(kind=kind)
            mo_typeindices = f.read_index()
            mo_vectors = cls.reshape_square(mo_vectors, n_bas)

            mo[kind] = OrbitalSet(mo_vectors,
                                  types=mo_typeindices,
                                  irreps=mo_irreps,
                                  energies=mo_energies,
                                  occupations=mo_occupations)

        return cls(mo, None, n_sym=n_sym, n_bas=n_bas)

    @staticmethod
    def reshape_square(arr, dims):
        """
        Return a block-diagonal array where the blocks are constructed from a
        flat input array and an array of dimensions for each block. The array
        is assumed to be layed out in memory using Fortran indexing.
        """
        if len(dims) == 1:
            dim = dims[0]
            return arr.reshape((dims[0],dims[0]), order='F')
        lst = []
        offset = 0
        for dim in dims:
            slice_ = arr[offset:offset+dim**2]
            lst.append(slice_.reshape((dim,dim), order='F'))
            offset += dim**2
        return la.block_diag(*lst)
