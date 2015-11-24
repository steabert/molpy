import numpy as np
from collections import namedtuple
from scipy import linalg as la
import re

from . import export
from .mh5 import MolcasHDF5
from .inporb import MolcasINPORB
from .tools import lst_to_arr, argsort
from .errors import Error, DataNotAvailable

typename = {
    'f': 'fro',
    'i': 'ina',
    '1': 'RAS1',
    '2': 'RAS2',
    '3': 'RAS3',
    's': 'sec',
    'd': 'del',
    '-': 'NA',
    }

angmom_name = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n']


@export
class BasisSet():
    """
    The BasisSet class groups data ragarding centers (their position and charge),
    and contains a array of contracted and primitive basis function IDs.
    """
    def __init__(self, center_labels, center_charges, center_coordinates,
                 contracted_ids, primitive_ids, primitives):
        self.center_labels = np.asarray(center_labels)
        self.center_charges = np.asarray(center_charges)
        self.center_coordinates = np.asarray(center_coordinates)
        self.contracted_ids = contracted_ids
        self.primitive_ids = primitive_ids
        self.primitives = primitives

        if self.contracted_ids is not None:
            center_ids = sorted(set((id[0] for id in self.contracted_ids)))
            assert len(self.center_labels) == len(center_ids)
            # assert center_ids[0] == 0
            # assert center_ids[-1] == len(self.centers) - 1

        self.n_cgto = self.contracted_ids.shape[0]
        self.n_pgto = self.primitive_ids.shape[0]

        self.cgto_molcas_indices = argsort(self._idtuples_ladder_order)
        self.cgto_molden_indices = argsort(self._idtuples_updown_order)

    @property
    def primitive_tree(self):
        """
        Generate a primitives hierarchy as a nested list of centers, angular momenta,
        and shells, where each shell is a dict of exponents and coefficients.

        Example code for printing the tree 'centers':

        """

        if self.primitive_ids is None or self.primitives is None:
            return None

        centers = []

        center_ids = self.primitive_ids[:,0]
        angmom_ids = self.primitive_ids[:,1]
        shell_ids = self.primitive_ids[:,2]

        for center_id in np.unique(center_ids):
            print(center_id, self.center_labels[center_id-1])
            center_selection = (center_ids == center_id)

            center = {}
            center['id'] = center_id
            center['label'] = self.center_labels[center_id-1]
            center['angmoms'] = []

            for angmom_id in np.unique(angmom_ids[center_selection]):
                angmom_selection = center_selection & (angmom_ids == angmom_id)

                angmom = {}
                angmom['value'] = angmom_id
                angmom['shells'] = []

                for shell_id in np.unique(shell_ids[angmom_selection]):
                    shell_selection = angmom_selection & (shell_ids == shell_id)

                    shell = {}
                    shell['id'] = shell_id
                    shell['exponents'] = self.primitives[shell_selection,0]
                    shell['coefficients'] = self.primitives[shell_selection,1]

                    angmom['shells'].append(shell)

                center['angmoms'].append(angmom)

            centers.append(center)
        return centers

    def copy(self):
        return BasisSet(
            self.center_labels.copy(),
            self.center_charges.copy(),
            self.center_coordinates.copy(),
            self.contracted_ids.copy(),
            self.primitive_ids.copy(),
            self.primitives.copy(),
            )

    def __getitem__(self, index):
        return BasisSet(
            self.center_labels.copy(),
            self.center_charges.copy(),
            self.center_coordinates.copy(),
            self.contracted_ids[index],
            self.primitive_ids.copy(),
            self.primitives.copy(),
            )

    def __str__(self):
        lines = []
        for center in self.primitive_tree:
            lines.append('center {:d} ({:s}):'.format(center['id'], center['label']))
            for angmom in center['angmoms']:
                lines.append('l = {:d}:'.format(angmom['value']))
                for shell in angmom['shells']:
                    lines.append('n = {:d}'.format(shell['id']))
                    coef = shell['coefficients']
                    exp = shell['exponents']
                    for c, e in zip(coef, exp):
                        lines.append('coef/exp = {:f} {:f}'.format(c, e))
        return '\n'.join(lines)

    @property
    def labels(self):
        """ convert basis function ID tuples into a human-readable label """
        label_list = []
        for basis_id in self.contracted_ids:
            c, n, l, m, = basis_id
            center_lbl = self.center_labels[c-1]
            n_lbl = str(n+l) if n+l < 10 else '*'
            l_lbl = angmom_name[l]
            if l == 0:
                m_lbl = ''
            elif l == 1:
                m_lbl = ['y','z','x'][m+1]
            else:
                m_lbl = str(m)[::-1]
            label =  '{:6s}{:1s}{:1s}{:2s}'.format(center_lbl, n_lbl, l_lbl, m_lbl)
            label_list.append(label)
        return np.array(label_list, dtype='U')

    @property
    def _idtuples_ladder_order(self):
        """
        rank a basis tuple according to Molcas ordering

        angmom components are ranked as (...,-2,-1,0,+1,+2,...)
        """
        idtuples = []
        for id_tuple in self.contracted_ids:
            center, n, l, m, = id_tuple
            if l == 1 and m == 1:
                m = -2
            idtuples.append((center, l, m, n))
        return idtuples

    @property
    def _idtuples_updown_order(self):
        """
        rank a basis tuple according to Molden/Gaussian ordering

        angmom components are ranked as (0,1+,1-, 2+,2-,...)
        """
        idtuples = []
        for id_tuple in self.contracted_ids:
            center, n, l, m, = id_tuple
            if l == 1 and m == 0:
                m = 2
            if m < 0:
                m = -m + 0.5
            idtuples.append((center, l, n, m))
        return idtuples

    def argsort_ids(self, ids=None, order='molcas'):
        """
        Reorder the supplied ids of the contracted functions by either
        Molcas or Molden/Gaussian ranking and return an array of indices.
        """

        if ids is None:
            ids = np.arange(self.n_cgto)

        if order == 'molcas':
            return self.cgto_molcas_indices[ids]
        elif order == 'molden':
            return self.cgto_molden_indices[ids]
        else:
            raise Error('invalid order parameter')


@export
class OrbitalSet():
    """
    Represents a set of orbitals with a common basis set, and keeps track of
    their properties (energy, occupation, type, irrep).
    """
    def __init__(self, coefficients, ids=None, types=None,
                 irreps=None, energies=None, occupations=None,
                 basis_ids=None, basis_set=None):

        self.coefficients = np.asmatrix(coefficients)
        self.n_bas = coefficients.shape[0]
        self.n_orb = coefficients.shape[1]
        assert self.n_bas != 0
        assert self.n_orb != 0

        if irreps is None:
            self.irreps = np.zeros(self.n_bas)
            self.n_irreps = 1
        else:
            self.irreps = irreps
            self.n_irreps = len(set(irreps))

        self.n_irreps = len(np.unique(irreps))

        if ids is None:
            self.ids = 1 + np.arange(self.n_orb)
        else:
            self.ids = ids

        if types is None:
            self.types = np.array(['-'] * self.n_bas)
        else:
            self.types = types

        if energies is None:
            self.energies = np.array([np.nan] * self.n_bas)
        else:
            self.energies = energies

        if occupations is None:
            self.occupations = np.array([np.nan] * self.n_bas)
        else:
            self.occupations = occupations

        if basis_ids is None:
            self.basis_ids = np.arange(self.n_bas)
        else:
            self.basis_ids = basis_ids

        self.basis_set = basis_set

    def copy(self):
        return self.__class__(
            self.coefficients.copy(),
            ids=self.ids.copy(),
            types=self.types.copy(),
            irreps=self.irreps.copy(),
            energies=self.energies.copy(),
            occupations=self.occupations.copy(),
            basis_ids=self.basis_ids.copy(),
            basis_set=self.basis_set,
            )

    def __getitem__(self, index):
        return self.__class__(
            self.coefficients[:,index],
            ids=self.ids[index],
            types=self.types[index],
            irreps=self.irreps[index],
            energies=self.energies[index],
            occupations=self.occupations[index],
            basis_ids=self.basis_ids.copy(),
            basis_set=self.basis_set,
            )

    def filter_basis(self, index):
        return self.__class__(
            self.coefficients[index,:],
            ids=self.ids.copy(),
            types=self.types.copy(),
            irreps=self.irreps.copy(),
            energies=self.energies.copy(),
            occupations=self.occupations.copy(),
            basis_ids=self.basis_ids[index],
            basis_set=self.basis_set,
            )

    def sort_basis(self, order='molcas'):

        ids = self.basis_set.argsort_ids(self.basis_ids, order=order)
        return self.filter_basis(ids)

    def __str__(self):
        """
        returns the Orbital coefficients formatted as columns
        """

        lines = []

        prefix = '{:16s}'
        int_template = prefix + self.n_orb * '{:10d}'
        float_template = prefix + self.n_orb * '{:10.4f}'
        str_template = prefix + self.n_orb * '{:>10s}'

        line = int_template.format("ID", *self.ids)
        lines.append(line)
        lines.append('')

        line = int_template.format("irrep", *self.irreps)
        lines.append(line)

        line = float_template.format('Occupation', *self.occupations)
        lines.append(line)

        line = float_template.format('Energy', *self.energies)
        lines.append(line)

        line = str_template.format('Type Index', *[typename[idx] for idx in self.types])
        lines.append(line)

        try:
            labels = self.basis_set.labels[self.basis_ids]
        except AttributeError:
            labels = self.basis_ids.astype('U')

        lines.append('')

        for ibas in range(self.n_bas):
            line = float_template.format(labels[ibas], *np.ravel(self.coefficients[ibas,:]))
            lines.append(line)

        return '\n'.join(lines)

    def show(self, cols=10):
        """
        prints the entire orbital set in blocks of cols orbitals
        """

        for offset in range(0, self.n_orb, cols):
            orbitals = self[offset:offset+cols]
            print(orbitals)

    def show_by_irrep(self, cols=10):

        if self.n_irreps > 1:
            for irrep in range(self.n_irreps):
                print('symmetry {:d}'.format(irrep))
                print()
                indices, = np.where(self.irreps == irrep)
                self[indices].sorted(reindex=True).show(cols=cols)
        else:
            self.show(cols=cols)

    def sorted(self, reindex=False):
        """
        returns a new orbitals set sorted first by typeid, then by
        increasing energy, and finally by decreasing occupation.
        """

        index = np.lexsort((self.energies, -self.occupations))

        if reindex:
            ids = None
        else:
            ids = self.ids[index]

        return self.__class__(
            self.coefficients[:,index],
            ids=ids,
            types=self.types[index],
            irreps=self.irreps[index],
            energies=self.energies[index],
            occupations=self.occupations[index],
            basis_ids=self.basis_ids.copy(),
            basis_set=self.basis_set,
            )

    def type(self, *typeids):
        """
        returns a new orbital set with only the requested type ids.
        """

        mo_indices = []
        for typeid in typeids:
            mo_set, = np.where(self.types == typeid)
            mo_indices.extend(mo_set)

        return self[mo_indices]

    def erange(self, lo, hi):
        """
        returns a new orbital set with orbitals that have an energy
        between lo and hi.
        """

        return self[(self.energies > lo) & (self.energies < hi)]

    def pattern(self, regex):
        """
        returns a new orbital set where the basis functions have been
        filtered as those which labels are matching the supplied regex.
        """
        matching = [bool(re.search(regex, label)) for label in self.basis_set.labels]

        return self.filter_basis(np.asarray(matching))


@export
class Wavefunction():
    def __init__(self, mo, basis_set, salcs=None, overlap=None, fockint=None, spinmult=None):
        self.mo = mo
        self.basis_set = basis_set
        self.salcs = salcs
        self.overlap = overlap
        self.fockint = fockint
        self.spinmult = spinmult

    def electronic_info(self):
        '''
        return a tuple containing the total number of electrons, the number of
        alpha electrons, the number of beta electrons, and the spin multiplicity.
        '''
        if 'alfa' in self.mo and 'beta' in self.mo:
            n_alfa = int(np.sum(self.mo['alfa'].occupations))
            n_beta = int(np.sum(self.mo['beta'].occupations))
            n_electrons = n_alfa + n_beta
            if self.spinmult is None:
                spinmult = n_alfa - n_beta + 1
            else:
                spinmult = self.spinmult
        elif 'restricted' in self.mo:
            try:
                n_electrons = int(np.sum(self.mo['restricted'].occupations))
            except ValueError:
                n_electrons = 0
            if self.spinmult is None:
                spinmult = 1
            else:
                spinmult = self.spinmult
            n_beta = (n_electrons - (spinmult - 1)) // 2
            n_alfa = n_electrons - n_beta
        else:
            raise InvalidRequest('orbital dict does not contain valid keys')
        electronic_charge = -n_electrons
        return (n_electrons, n_alfa, n_beta, spinmult, electronic_charge)

    def nuclear_info(self):
        '''
        return a tuple containing the total number of atoms and nuclear charge
        '''
        n_atoms = len(self.basis_set.center_labels)
        nuclear_charge = int(np.sum(self.basis_set.center_charges))
        return (n_atoms, nuclear_charge)

    def print_orbitals(self, desym=False, types=None, erange=None, pattern=None,
                       kind='restricted', order=None):

        try:
            orbitals = self.mo[kind]
        except KeyError:
            raise Error('invalid orbital kind parameter')

        if types is not None:
            orbitals = orbitals.type(*types)

        if erange is not None:
            orbitals = orbitals.erange(*erange)

        if pattern is not None:
            orbitals = orbitals.pattern(pattern)

        if order is not None:
            orbitals = orbitals.sort_basis(order=order)

        if desym:
            orbitals.show()
        else:
            orbitals.show_by_irrep()

    def guessorb(self, kind='restricted'):
        """
        generate a set of initial molecular orbitals
        """
        guessorb = {}
        Smat_ao = np.asmatrix(self.overlap)
        Fmat_ao = np.asmatrix(self.fockint)
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
                Fmat_mo = Cmat.T * Smat_ao.T * Fmat_ao * Smat_ao * Cmat
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
        return Wavefunction(guessorb, self.basis_set)

    @classmethod
    def from_h5(cls, filename):
        """ Generates a wavefunction from a Molcas HDF5 file """
        f = MolcasHDF5(filename, 'r')

        n_bas = f.n_bas

        if n_bas is None:
            raise Exception('no basis set size available on file')

        n_irreps = len(n_bas)

        if n_irreps > 1:
            n_atoms = f.natoms_all()
            center_labels = f.desym_center_labels()
            center_charges = f.desym_center_charges()
            center_coordinates = f.desym_center_coordinates()
            contracted_ids = f.desym_basis_function_ids()
            salcs = f.desym_matrix()
        else:
            n_atoms = f.natoms_unique()
            center_labels = f.center_labels()
            center_charges = f.center_charges()
            center_coordinates = f.center_coordinates()
            contracted_ids = f.basis_function_ids()
            overlap = f.ao_overlap_matrix()
            fockint = f.ao_fockint_matrix()
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

        if n_irreps > 1:
            irrep_list = list(np.array([irrep]*nb) for irrep, nb in enumerate(n_bas))
            mo_irreps = lst_to_arr(irrep_list)
        else:
            mo_irreps = lst_to_arr(f.supsym_irrep_indices())

        unrestricted = f.unrestricted()
        if unrestricted:
            kinds = ['alfa', 'beta']
        else:
            kinds = ['restricted']

        mo = {}
        for kind in kinds:
            mo_occupations = f.mo_occupations(kind=kind)
            mo_energies = f.mo_energies(kind=kind)
            mo_typeindices = f.mo_typeindices(kind=kind)
            mo_vectors = f.mo_vectors(kind=kind)

            mo_occupations = lst_to_arr(mo_occupations)
            mo_energies = lst_to_arr(mo_energies)
            mo_typeindices = lst_to_arr(mo_typeindices)
            mo_vectors = la.block_diag(*mo_vectors)

            if n_irreps > 1:
                mo_vectors = np.dot(salcs, mo_vectors)

            mo[kind] = OrbitalSet(mo_vectors,
                                  types=mo_typeindices,
                                  irreps=mo_irreps,
                                  energies=mo_energies,
                                  occupations=mo_occupations,
                                  basis_set=basis_set)

        overlap = la.block_diag(*f.ao_overlap_matrix())
        fockint = la.block_diag(*f.ao_fockint_matrix())
        if n_irreps > 1:
            overlap = np.dot(np.dot(salcs, overlap), salcs.T)
            fockint = np.dot(np.dot(salcs, fockint), salcs.T)

        try:
            ispin = f.ispin()
        except DataNotAvailable:
            ispin = None

        return cls(mo, basis_set,
                   overlap=overlap, fockint=fockint,
                   spinmult=ispin)

    @classmethod
    def from_inporb(cls, filename):
        """ Generates a wavefunction from a Molcas INPORB file """
        f = inporb.MolcasINPORB(filename, 'r')

        n_bas = f.n_bas

        if n_bas is None:
            raise Exception('no basis set size available on file')

        n_irreps = len(n_bas)
