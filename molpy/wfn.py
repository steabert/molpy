from . import export
from . import mh5
import numpy as np
from collections import namedtuple
from .tools import lst_to_arr, argsort
from scipy import linalg as la
from .errors import Error, DataNotAvailable
import re

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
            center_selection = (center_ids == center_id)

            center = {}
            center['id'] = center_id
            center['label'] = self.center_labels[center_id]
            center['angmoms'] = []

            for angmom_id in np.unique(angmom_ids):
                angmom_selection = center_selection & (angmom_ids == angmom_id)

                angmom = {}
                angmom['value'] = angmom_id
                angmom['shells'] = []

                for shell_id in np.unique(shell_ids):
                    shell_selection = angmom_selection & (shell_ids == shell_id)

                    shell = {}
                    shell['id'] = shell_id
                    shell['exponents'] = primitives[shell_selection,0]
                    shell['coefficients'] = primitives[shell_selection,1]

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
            return np.argsort(self.cgto_molcas_indices[ids])
        elif order == 'molden':
            return np.argsort(self.cgto_molden_indices[ids])
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
            if self.n_irreps > 1:
                self.ids = np.zeros(self.n_orb, dtype=int)
                for irrep in range(self.n_irreps):
                    mo_set, = np.where(self.irreps == irrep)
                    self.ids[mo_set] = 1 + np.arange(len(mo_set))
            else:
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
        int_template = prefix + self.n_orb * '{:10d}' + '\n'
        float_template = prefix + self.n_orb * '{:10.4f}' + '\n'
        str_template = prefix + self.n_orb * '{:>10s}' + '\n'

        line = str_template.format("MO ID", *[str(t) for t in zip(self.irreps, self.ids)])
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

        lines.append('\n')

        for ibas in range(self.n_bas):
            line = float_template.format(labels[ibas], *np.ravel(self.coefficients[ibas,:]))
            lines.append(line)

        return ''.join(lines)

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
                self[indices].sorted().show(cols=cols)
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
    def __init__(self, orbitals=None, salcs=None, overlap=None, density=None):
        try:
            self.mos_alpha, self.mos_beta = orbitals
            self.uhf = True
        except:
            self.mos_restricted = orbitals
            self.uhf = False
        self.salcs = salcs

    def print_orbitals(self, by_irrep=True, types=None, erange=None, pattern=None,
                       kind='restricted', order='molcas'):

        if kind == 'restricted':
            orbitals = self.mos_restricted
        elif kind == 'alpha':
            orbitals = self.mos_alpha
        elif kind == 'beta':
            orbitals = self.mos_beta
        else:
            raise Error('invalid orbital kind parameter')

        if types is not None:
            orbitals = orbitals.type(*types)

        if erange is not None:
            orbitals = orbitals.erange(*erange)

        if pattern is not None:
            orbitals = orbitals.pattern(pattern)

        orbitals = orbitals.sort_basis(order=order)

        if by_irrep:
            orbitals.show_by_irrep()
        else:
            orbitals.show()

@export
def gen_from_mh5(filename):
    """ Generates a wavefunction from a Molcas HDF5 file """
    f = mh5.MolcasHDF5(filename, 'r')

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
        mo = {}
        for kind in ('alpha', 'beta'):
            mo_occupations = f.mo_occupations(kind=kind)
            mo_energies = f.mo_energies(kind=kind)
            mo_typeindices = f.mo_typeindices(kind=kind)
            mo_vectors = f.mo_vectors(kind=kind)

            mo_occupations = lst_to_arr(mo_occupations)
            mo_energies = lst_to_arr(mo_energies)
            mo_typeindices = lst_to_arr(mo_typeindices)
            mo_vectors = la.block_diag(*mo_vectors)

            mo[kind] = OrbitalSet(mo_vectors,
                                  types=mo_typeindices,
                                  irreps=mo_irreps,
                                  energies=mo_energies,
                                  occupations=mo_occupations,
                                  basis_set=basis_set)
        mo_alpha = mo['alpha']
        mo_beta = mo['beta']
    else:
        mo_occupations = f.mo_occupations()
        mo_energies = f.mo_energies()
        mo_typeindices = f.mo_typeindices()
        mo_vectors = f.mo_vectors()

        mo_occupations = lst_to_arr(mo_occupations)
        mo_energies = lst_to_arr(mo_energies)
        mo_typeindices = lst_to_arr(mo_typeindices)
        mo_vectors = la.block_diag(*mo_vectors)

        mo = OrbitalSet(mo_vectors,
                        types=mo_typeindices,
                        irreps=mo_irreps,
                        energies=mo_energies,
                        occupations=mo_occupations,
                        basis_set=basis_set)

    return Wavefunction(orbitals=mo)
