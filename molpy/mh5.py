from . import export
from .tools import *
from .errors import InvalidRequest, DataNotAvailable
import h5py
import numpy as np


@export
class MolcasHDF5:

    def __init__(self, filename, mode):
        """ initialize the HDF5 file object """
        self.h5f = h5py.File(filename, mode)

        if mode.startswith('r'):
            try:
                self.n_sym = self.h5f.attrs['NSYM']
                self.n_bas = self.h5f.attrs['NBAS']
            except:
                raise DataNotAvailable('required data is missing!')
            assert self.n_sym == len(self.n_bas)

    def maybe_fetch_attr(self, field):

        try:
            data = self.h5f.attrs[field]
        except:
            raise DataNotAvailable('HDF5 file is missing field {:s}'.format(field))
        return data

    def maybe_fetch_dset(self, field):

        try:
            data = np.array(self.h5f[field])
        except:
            raise DataNotAvailable('HDF5 file is missing field {:s}'.format(field))
        return data

    def molcas_version(self):

        version_bytes = self.maybe_fetch_attr('MOLCAS_VERSION')
        if version_bytes is not None:
            version = version_bytes.decode()
        else:
            version = 'unknown'
        return version

    def molcas_module(self):

        module_bytes = self.maybe_fetch_attr('MOLCAS_MODULE')
        if module_bytes is not None:
            module = module_bytes.decode()
        else:
            module = 'unknown'
        return module

    def irrep_labels(self):
        data_bytes = self.maybe_fetch_attr('IRREP_LABELS')
        if data_bytes is not None:
            labels = np.array(data_bytes, dtype='U')
        else:
            labels = None
        return labels

    def natoms_unique(self):
        return self.maybe_fetch_attr('NATOMS_UNIQUE')

    def center_labels(self):
        data_bytes = self.maybe_fetch_dset('CENTER_LABELS')
        if data_bytes is not None:
            labels = np.array(list(data_bytes), dtype='U')
        else:
            labels = None
        return labels

    def center_charges(self):
        return self.maybe_fetch_dset('CENTER_CHARGES')

    def center_coordinates(self):
        return self.maybe_fetch_dset('CENTER_COORDINATES')

    def basis_function_ids(self):

        data = self.maybe_fetch_dset('BASIS_FUNCTION_IDS')
        if data is not None:
            ids = arr_to_lst(data.flatten(), [(4,nb) for nb in self.n_bas])
            ids = [arr.T for arr in ids]
        else:
            ids = None
        return ids

    def natoms_all(self):
        return self.maybe_fetch_attr('NATOMS_ALL')

    def desym_center_labels(self):

        attribute = 'DESYM_CENTER_LABELS'
        data_bytes = self.maybe_fetch_dset(attribute)
        return np.array(list(data_bytes), dtype='U')

    def desym_center_charges(self):
        return self.maybe_fetch_dset('DESYM_CENTER_CHARGES')

    def desym_center_coordinates(self):
        return self.maybe_fetch_dset('DESYM_CENTER_COORDINATES')

    def desym_basis_function_ids(self):
        return self.maybe_fetch_dset('DESYM_BASIS_FUNCTION_IDS')

    def desym_matrix(self):

        data_flat = self.maybe_fetch_dset('DESYM_MATRIX')
        if data_flat is not None:
            self.n_bast = np.sum(self.n_bas)
            desym_matrix = data_flat[:].reshape((self.n_bast,self.n_bast), order='F')
        else:
            desym_matrix = None
        return desym_matrix

    def primitives(self):
        return self.maybe_fetch_dset('PRIMITIVES')

    def primitive_ids(self):
        return self.maybe_fetch_dset('PRIMITIVE_IDS')

    def basisset(self):
        """
        generate a basisset as a nested list of centers, angular momenta,
        and shells, where each shell is a dict of exponents and coefficients
        """
        nprim = self.maybe_fetch_attr('NPRIM')

        if nprim is None:
            return None

        primids = self.maybe_fetch_dset('PRIMITIVE_IDS')
        prims = self.maybe_fetch_dset('PRIMITIVES')

        if primids is None or prims is None:
            return None

        basisset = []

        center_ids = primids[:,0]
        centers = np.unique(center_ids)
        for center in centers:
            index_center, = np.where(center_ids == center)
            primids_center = primids[index_center,:]
            prims_center = prims[index_center,:]

            basisset_center = {}
            basisset_center['id'] = center
            basisset_center['angmoms'] = []

            angmom_ids = primids_center[:,1]
            angmoms = np.unique(angmom_ids)
            for angmom in angmoms:
                index_angmom, = np.where(angmom_ids == angmom)
                primids_angmom = primids_center[index_angmom,:]
                prims_angmom = prims_center[index_angmom,:]

                basisset_angmom = {}
                basisset_angmom['value'] = angmom
                basisset_angmom['shells'] = []

                shell_ids = primids_angmom[:,2]
                shells = np.unique(shell_ids)
                for shell in shells:
                    index_shell, = np.where(shell_ids == shell)

                    basisset_shell = {}
                    basisset_shell['id'] = shell
                    basisset_shell['exponents'] = prims_angmom[index_shell,0]
                    basisset_shell['coefficients'] = prims_angmom[index_shell,1]

                    basisset_angmom['shells'].append(basisset_shell)

                basisset_center['angmoms'].append(basisset_angmom)

            basisset.append(basisset_center)

        return basisset

    def ispin(self):
        """ obtain spin multiplicity """
        return self.maybe_fetch_attr('ISPIN')

    def unrestricted(self):
        try:
            orbital_type = self.maybe_fetch_attr('ORBITAL_TYPE')
            if orbital_type.decode().endswith('UHF'):
                is_unrestructed = True
            else:
                is_unrestricted = False
        except DataNotAvailable:
            is_unrestricted = False
        return is_unrestricted


    def _get_mo_attribute(self, attribute, kind='restricted'):
        """
        Converts an MO attribute to an attribute with a specific kind.
        If a certain kind conflicts with the wavefunction type, an
        InvalidRequest exception is raised.
        """

        if kind == 'restricted' and self.unrestricted():
            raise InvalidRequest('UHF wavefunction has no restricted orbitals')
        elif kind != 'restricted' and not self.unrestricted():
            raise InvalidRequest('RHF wavefunction has no alpha/beta orbitals')

        attribute_prefix = 'MO_'
        if kind == 'alpha':
            attribute_prefix += 'ALPHA_'
        elif kind == 'beta':
            attribute_prefix += 'BETA_'

        return attribute_prefix + attribute

    def mo_typeindices(self, kind='restricted'):

        attribute = self._get_mo_attribute('TYPEINDICES', kind=kind)
        try:
            data_bytes = self.maybe_fetch_dset(attribute)
            typeindices = np.char.lower(np.array(list(data_bytes), dtype='U'))
        except DataNotAvailable:
            typeindices = np.array(['u'] * sum(self.n_bas), dtype='U')
        return arr_to_lst(typeindices, self.n_bas)

    def mo_occupations(self, kind='restricted'):

        attribute = self._get_mo_attribute('OCCUPATIONS', kind=kind)
        try:
            occupations = self.maybe_fetch_dset(attribute)
        except DataNotAvailable:
            occupations = np.empty(sum(self.n_bas), dtype=float64)
            occupations.fill(np.nan)
        return arr_to_lst(occupations, self.n_bas)

    def mo_energies(self, kind='restricted'):

        attribute = self._get_mo_attribute('ENERGIES', kind=kind)
        try:
            energies = self.maybe_fetch_dset(attribute)
        except DataNotAvailable:
            energies = np.empty(sum(self.n_bas), dtype=float64)
            energies.fill(np.nan)
        return arr_to_lst(energies, self.n_bas)

    def mo_vectors(self, kind='restricted'):

        attribute = self._get_mo_attribute('VECTORS', kind=kind)
        try:
            vectors = self.maybe_fetch_dset(attribute)
        except DataNotAvailable:
            vectors = np.empty(sum(self.n_bas**2), dtype=float64)
            vectors.fill(np.nan)
        return arr_to_lst(vectors, [(nb,nb) for nb in self.n_bas])

    def ao_overlap_matrix(self):

        attribute = 'AO_OVERLAP_MATRIX'
        overlap_matrix = self.maybe_fetch_dset(attribute)
        return arr_to_lst(overlap_matrix, [(nb,nb) for nb in self.n_bas])

    def ao_fockint_matrix(self):

        attribute = 'AO_FOCKINT_MATRIX'
        fockint_matrix = self.maybe_fetch_dset(attribute)
        return arr_to_lst(fockint_matrix, [(nb,nb) for nb in self.n_bas])

    def supsym_irrep_indices(self):

        data_flat = self.maybe_fetch_dset('SUPSYM_IRREP_INDICES')
        if data_flat is not None:
            if not self.uhf:
                rhf = arr_to_lst(data_flat, self.n_bas)
                indices = (rhf, None)
            else:
                self.n_bast = np.sum(self.n_bas)
                alpha = arr_to_lst(data_flat[:self.n_bast], self.n_bas)
                beta = arr_to_lst(data_flat[self.n_bast:], self.n_bas)
                indices = (alpha, beta)
        else:
            indices = (None, None)
        return indices

    def supsym_irrep_labels(self):
        data_bytes = self.maybe_fetch_dset('SUPSYM_IRREP_LABELS')
        if data_bytes is not None:
            labels = np.array(list(data_bytes), dtype='U')
        else:
            labels = None
        return labels

    def write(self, wfn):
        self.h5f.attrs['NSYM'] = wfn.nsym
        self.h5f.attrs['self.n_bas'] = lst_to_arr(wfn.self.n_bas)
        self.h5f.attrs['IRREP_LABELS'] = wfn.irrep_labels.astype('S')
        self.h5f['MO_VECTORS'] = lst_to_arr(wfn.mo_vectors)
        # self.h5f['MO_OCCUPATIONS'] = lst_to_arr(wfn.mo_occupations)
        self.h5f['MO_ENERGIES'] = lst_to_arr(wfn.mo_energies)
        # self.h5f['MO_TYPEINDICES'] = lst_to_arr(wfn.mo_typeindices).astype('S')
        self.h5f['SUPSYM_IRREP_INDICES'] = lst_to_arr(wfn.supsym_irrep_indices)
        self.h5f['SUPSYM_IRREP_LABELS'] = lst_to_arr(wfn.supsym_irrep_labels).astype('S8')
