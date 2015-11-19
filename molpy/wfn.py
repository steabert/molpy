from . import export
from . import mh5
import numpy as np
from collections import namedtuple
from .tools import lst_to_arr
from scipy import linalg as la
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
    def __init__(self, centers=None, charges=None, coordinates=None,
                 contracted_ids=None, primitive_ids=None, primitives=None):
        self.centers = centers
        self.charges = charges
        self.coordinates = coordinates
        self.contracted_ids = contracted_ids
        self.primitive_ids = primitive_ids
        self.primitives = primitives

        if self.contracted_ids is not None:
            center_ids = sorted(set((id[0] for id in self.contracted_ids)))
            assert len(self.centers) == len(center_ids)
            # assert center_ids[0] == 0
            # assert center_ids[-1] == len(self.centers) - 1

        self._build_primitive_hierarchy()

    def _build_primitive_hierarchy(self):
        """
        generate a primitives hierarchy as a nested list of centers, angular momenta,
        and shells, where each shell is a dict of exponents and coefficients
        """

        if self.primitive_ids is None or self.primitives is None:
            self.primitive_tree = None
            return

        self.primitive_tree = []

        center_ids = self.primitive_ids[:,0]
        centers = np.unique(center_ids)
        for center in centers:
            index_center, = np.where(center_ids == center)
            primitive_ids_center = self.primitive_ids[index_center,:]
            prims_center = self.primitives[index_center,:]

            primitive_tree_center = {}
            primitive_tree_center['id'] = center
            primitive_tree_center['angmoms'] = []

            angmom_ids = primitive_ids_center[:,1]
            angmoms = np.unique(angmom_ids)
            for angmom in angmoms:
                index_angmom, = np.where(angmom_ids == angmom)
                primitive_ids_angmom = primitive_ids_center[index_angmom,:]
                prims_angmom = prims_center[index_angmom,:]

                primitive_tree_angmom = {}
                primitive_tree_angmom['value'] = angmom
                primitive_tree_angmom['shells'] = []

                shell_ids = primitive_ids_angmom[:,2]
                shells = np.unique(shell_ids)
                for shell in shells:
                    index_shell, = np.where(shell_ids == shell)

                    primitive_tree_shell = {}
                    primitive_tree_shell['id'] = shell
                    primitive_tree_shell['exponents'] = prims_angmom[index_shell,0]
                    primitive_tree_shell['coefficients'] = prims_angmom[index_shell,1]

                    primitive_tree_angmom['shells'].append(primitive_tree_shell)

                primitive_tree_center['angmoms'].append(primitive_tree_angmom)

            self.primitive_tree.append(primitive_tree_center)
        return

    def copy(self):
        return BasisSet(
            list(self.centers),
            list(self.charges),
            list(self.coordinates),
            self.contracted_ids.copy(),
            self.primitive_ids.copy(),
            self.primitives.copy(),
            )

    def __getitem__(self, index):
        return BasisSet(
            list(self.centers),
            list(self.charges),
            list(self.coordinates),
            self.contracted_ids[index],
            self.primitive_ids.copy(),
            self.primitives.copy(),
            )

    @property
    def labels(self):
        """ convert basis function ID tuples into a human-readable label """
        label_list = []
        for basis_id in self.contracted_ids:
            c, n, l, m, = basis_id
            center_lbl = self.centers[c-1]
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
        return label_list


@export
class OrbitalSet():
    """
    Represents a set of orbitals with a common basis set, and keeps track of
    their properties (energy, occupation, type, irrep).
    """
    def __init__(self, coefficients, basis_set=None, types=None,
                 irreps=None, energies=None, occupations=None,
                 indices=None):
        self.coefficients = np.asmatrix(coefficients)
        self.n_bas = coefficients.shape[0]
        self.n_orb = coefficients.shape[1]
        assert self.n_bas != 0
        assert self.n_orb != 0

        if basis_set is not None:
            if not isinstance(basis_set, BasisSet):
                raise TypeError('please provide basis_set=<BasisSet>')
            self.basis_set = basis_set
        else:
            self.basis_set = BasisSet()

        if types is None:
            self.types = np.array(['u'] * self.n_bas)
        else:
            self.types = types

        if irreps is None:
            self.irreps = np.zeros(self.n_bas)
            self.n_irreps = 1
        else:
            self.irreps = irreps
            self.n_irreps = len(set(irreps))

        if energies is None:
            self.energies = np.array([np.nan] * self.n_bas)
        else:
            self.energies = energies

        if occupations is None:
            self.occupations = np.array([np.nan] * self.n_bas)
        else:
            self.occupations = occupations

        if indices is None:
            if self.n_irreps > 1:
                self.indices = np.zeros(self.n_orb, dtype=int)
                for irrep in range(self.n_irreps):
                    mo_set, = np.where(self.irreps == irrep)
                    self.indices[mo_set] = 1 + np.arange(len(mo_set))
            else:
                self.indices = 1 + np.arange(self.n_orb)
        else:
            self.indices = indices

    def copy(self):
        return self.__class__(
            self.coefficients.copy(),
            basis_set=self.basis_set.copy(),
            types=self.types.copy(),
            irreps=self.irreps.copy(),
            energies=self.energies.copy(),
            occupations=self.occupations.copy(),
            indices=self.indices.copy(),
            )

    def __getitem__(self, index):
        return self.__class__(
            self.coefficients[:,index],
            basis_set=self.basis_set.copy(),
            types=self.types[index],
            irreps=self.irreps[index],
            energies=self.energies[index],
            occupations=self.occupations[index],
            indices=self.indices[index],
            )

    def filter_basis(self, index):
        return self.__class__(
            self.coefficients[index,:],
            basis_set=self.basis_set[index],
            types=self.types.copy(),
            irreps=self.irreps.copy(),
            energies=self.energies.copy(),
            occupations=self.occupations.copy(),
            indices=self.indices.copy(),
            )

    def __str__(self):
        """
        returns the Orbital coefficients formatted as columns
        """

        lines = []

        prefix = '{:16s}'
        int_template = prefix + self.n_orb * '{:10d}' + '\n'
        float_template = prefix + self.n_orb * '{:10.4f}' + '\n'
        str_template = prefix + self.n_orb * '{:>10s}' + '\n'

        line = str_template.format("MO ID", *[str(t) for t in zip(self.irreps, self.indices)])
        lines.append(line)

        line = float_template.format('Occupation', *self.occupations)
        lines.append(line)

        line = float_template.format('Energy', *self.energies)
        lines.append(line)

        line = str_template.format('Type Index', *[typename[idx] for idx in self.types])
        lines.append(line)

        try:
            labels = self.basis_set.labels
        except AttributeError:
            labels = np.arange(self.n_bas).astype('U')

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
            indices = None
        else:
            indices = self.indices[index]

        return self.__class__(
            self.coefficients[:,index],
            basis_set=self.basis_set.copy(),
            types=self.types[index],
            irreps=self.irreps[index],
            energies=self.energies[index],
            occupations=self.occupations[index],
            indices=indices,
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

    def print_orbitals(self, by_irrep=True, types=None, erange=None, kind='restricted'):

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
            centers=center_labels,
            charges=center_charges,
            coordinates=center_coordinates,
            contracted_ids=contracted_ids,
            primitive_ids=primitive_ids,
            primitives=primitives
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
