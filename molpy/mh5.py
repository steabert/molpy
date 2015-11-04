from . import export
from .tools import *
import h5py
import numpy as np


@export
class MolcasHDF5:

    def __init__(self, filename, mode):
        """ initialize the HDF5 file object """
        self.h5f = h5py.File(filename, mode)

    def maybe_fetch_attr(self, field):

        try:
            data = self.h5f.attrs[field]
        except:
            warn('HDF5 file is missing field {:s}'.format(field))
            data = None
        return data

    def maybe_fetch_dset(self, field):

        try:
            data = np.array(self.h5f[field])
        except:
            warn('HDF5 file is missing field {:s}'.format(field))
            data = None
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

    def nsym(self):
        return self.maybe_fetch_attr('NSYM')

    def nbas(self):
        return self.maybe_fetch_attr('NBAS')

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

    def basis_function_ids(self, nbas):

        data = self.maybe_fetch_dset('BASIS_FUNCTION_IDS')
        if data is not None:
            ids = arr_to_lst(data.flatten(), [(4,nb) for nb in nbas])
            ids = [arr.T for arr in ids]
        else:
            ids = None
        return ids

    def natoms_all(self):
        return self.maybe_fetch_attr('NATOMS_ALL')

    def desym_center_labels(self):
        data_bytes = self.maybe_fetch_dset('DESYM_CENTER_LABELS')
        if data_bytes is not None:
            labels = np.array(list(data_bytes), dtype='U')
        else:
            labels = None
        return labels

    def desym_center_charges(self):
        return self.maybe_fetch_dset('DESYM_CENTER_CHARGES')

    def desym_center_coordinates(self):
        return self.maybe_fetch_dset('DESYM_CENTER_COORDINATES')

    def desym_basis_function_ids(self):
        return self.maybe_fetch_dset('DESYM_BASIS_FUNCTION_IDS')

    def desym_matrix(self, nbas):

        data_flat = self.maybe_fetch_dset('DESYM_MATRIX')
        if data_flat is not None:
            nbast = np.sum(nbas)
            desym_matrix = data_flat[:].reshape((nbast,nbast), order='F')
        else:
            desym_matrix = None
        return desym_matrix

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

    def uhf(self):
        mo_space = self.maybe_fetch_attr('ORBITAL_TYPE')
        if mo_space is not None and mo_space.decode().endswith('UHF'):
            return True
        else:
            return False

    def ispin(self):
        """ obtain spin multiplicity """
        return self.maybe_fetch_attr('ISPIN')

    def mo_typeindices(self, nbas, uhf):

        data_bytes = self.maybe_fetch_dset('MO_TYPEINDICES')
        if data_bytes is not None:
            data_flat = np.char.lower(np.array(list(data_bytes), dtype='U'))
            typeindices = arr_to_lst(data_flat, nbas)
            if not uhf:
                rhf = arr_to_lst(data_flat, nbas)
                typeindices = (rhf, None)
            else:
                nbast = np.sum(nbas)
                alpha = arr_to_lst(data_flat[:nbast], nbas)
                beta = arr_to_lst(data_flat[nbast:], nbas)
                typeindices = (alpha, beta)
        else:
            typeindices = (None, None)
        return typeindices

    def mo_occupations(self, nbas, uhf):

        data_flat = self.maybe_fetch_dset('MO_OCCUPATIONS')
        if data_flat is not None:
            if not uhf:
                rhf = arr_to_lst(data_flat, nbas)
                occupations = (rhf, None)
            else:
                nbast = np.sum(nbas)
                alpha = arr_to_lst(data_flat[:nbast], nbas)
                beta = arr_to_lst(data_flat[nbast:], nbas)
                occupations = (alpha, beta)
        else:
            occupations = (None, None)
        return occupations

    def mo_energies(self, nbas, uhf):

        data_flat = self.maybe_fetch_dset('MO_ENERGIES')
        if data_flat is not None:
            if not uhf:
                rhf = arr_to_lst(data_flat, nbas)
                energies = (rhf, None)
            else:
                nbast = np.sum(nbas)
                alpha = arr_to_lst(data_flat[:nbast], nbas)
                beta = arr_to_lst(data_flat[nbast:], nbas)
                energies = (alpha, beta)
        else:
            energies = (None, None)
        return energies

    def mo_vectors(self, nbas, uhf):

        data_flat = self.maybe_fetch_dset('MO_VECTORS')
        if data_flat is not None:
            if not uhf:
                rhf = arr_to_lst(data_flat, [(nb,nb) for nb in nbas])
                vectors = (rhf, None)
            else:
                nbast = np.sum(nbas**2)
                alpha = arr_to_lst(data_flat[:nbast], [(nb,nb) for nb in nbas])
                beta = arr_to_lst(data_flat[nbast:], [(nb,nb) for nb in nbas])
                vectors = (alpha, beta)
        else:
            vectors = (None, None)
        return vectors

    def ao_overlap_matrix(self, nbas):

        data_flat = self.maybe_fetch_dset('AO_OVERLAP_MATRIX')
        if data_flat is not None:
            ao_overlap = arr_to_lst(data_flat, [(nb,nb) for nb in nbas])
        else:
            ao_overlap = None
        return ao_overlap

    def ao_fockint_matrix(self, nbas):

        data_flat = self.maybe_fetch_dset('AO_FOCKINT_MATRIX')
        if data_flat is not None:
            ao_fockint = arr_to_lst(data_flat, [(nb,nb) for nb in nbas])
        else:
            ao_fockint = None
        return ao_fockint

    def supsym_irrep_indices(self, nbas, uhf):

        data_flat = self.maybe_fetch_dset('SUPSYM_IRREP_INDICES')
        if data_flat is not None:
            if not uhf:
                rhf = arr_to_lst(data_flat, nbas)
                indices = (rhf, None)
            else:
                nbast = np.sum(nbas)
                alpha = arr_to_lst(data_flat[:nbast], nbas)
                beta = arr_to_lst(data_flat[nbast:], nbas)
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
        self.h5f.attrs['NBAS'] = lst_to_arr(wfn.nbas)
        self.h5f.attrs['IRREP_LABELS'] = wfn.irrep_labels.astype('S')
        self.h5f['MO_VECTORS'] = lst_to_arr(wfn.mo_vectors)
        # self.h5f['MO_OCCUPATIONS'] = lst_to_arr(wfn.mo_occupations)
        self.h5f['MO_ENERGIES'] = lst_to_arr(wfn.mo_energies)
        # self.h5f['MO_TYPEINDICES'] = lst_to_arr(wfn.mo_typeindices).astype('S')
        self.h5f['SUPSYM_IRREP_INDICES'] = lst_to_arr(wfn.supsym_irrep_indices)
        self.h5f['SUPSYM_IRREP_LABELS'] = lst_to_arr(wfn.supsym_irrep_labels).astype('S8')
