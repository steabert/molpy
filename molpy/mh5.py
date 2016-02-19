# mh5.py -- Molcas HDF5 format
#
# molpy, an orbital analyzer and file converter for Molcas files
# Copyright (c) 2016  Steven Vancoillie
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Written by Steven Vancoillie.
#

import h5py
import numpy as np

from . import export
from .errors import InvalidRequest, DataNotAvailable


@export
class MolcasHDF5:

    def __init__(self, filename, mode):
        """ initialize the HDF5 file object """
        if mode.startswith('r') and not h5py.is_hdf5(filename):
            raise InvalidRequest

        self.h5f = h5py.File(filename, mode)

        if mode.startswith('r'):
            try:
                self.n_sym = self.h5f.attrs['NSYM']
                self.n_bas = self.h5f.attrs['NBAS']
            except:
                raise DataNotAvailable('requested data is missing on the HDF5 file')
            assert self.n_sym == len(self.n_bas)

    def close(self):
        self.h5f.close()

    def maybe_fetch_attr(self, field):

        try:
            data = self.h5f.attrs[field]
        except:
            raise DataNotAvailable('HDF5 missing attribute: {:s}'.format(field))
        return data

    def maybe_fetch_dset(self, field):

        try:
            data = np.array(self.h5f[field])
        except:
            raise DataNotAvailable('HDF5 missing dataset: {:s}'.format(field))
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

        return self.maybe_fetch_dset('BASIS_FUNCTION_IDS')

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

    def ispin(self):
        """ obtain spin multiplicity """
        return self.maybe_fetch_attr('ISPIN')

    def unrestricted(self):
        try:
            orbital_type = self.maybe_fetch_attr('ORBITAL_TYPE')
            if orbital_type.decode().endswith('UHF'):
                is_unrestricted = True
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
            data_bytes = np.asarray(self.maybe_fetch_dset(attribute), dtype='U')
            typeindices = np.char.lower(data_bytes)
        except DataNotAvailable:
            typeindices = np.array(['-'] * sum(self.n_bas), dtype='U')
        return typeindices

    def mo_occupations(self, kind='restricted'):

        attribute = self._get_mo_attribute('OCCUPATIONS', kind=kind)
        try:
            occupations = np.asarray(self.maybe_fetch_dset(attribute))
        except DataNotAvailable:
            occupations = np.empty(sum(self.n_bas), dtype=np.float64)
            occupations.fill(np.nan)
        return occupations

    def mo_energies(self, kind='restricted'):

        attribute = self._get_mo_attribute('ENERGIES', kind=kind)
        try:
            energies = np.asarray(self.maybe_fetch_dset(attribute))
        except DataNotAvailable:
            energies = np.empty(sum(self.n_bas), dtype=np.float64)
            energies.fill(np.nan)
        return energies

    def mo_vectors(self, kind='restricted'):

        attribute = self._get_mo_attribute('VECTORS', kind=kind)
        try:
            vectors = np.asarray(self.maybe_fetch_dset(attribute))
        except DataNotAvailable:
            vectors = np.empty(sum(self.n_bas**2), dtype=np.float64)
            vectors.fill(np.nan)
        return vectors

    def ao_overlap_matrix(self):

        attribute = 'AO_OVERLAP_MATRIX'
        return np.asarray(self.maybe_fetch_dset(attribute))

    def ao_fockint_matrix(self):

        attribute = 'AO_FOCKINT_MATRIX'
        return np.asarray(self.maybe_fetch_dset(attribute))

    def densities(self):

        attribute = 'DENSITY_MATRIX'
        return np.asarray(self.maybe_fetch_dset(attribute))

    def spindens(self):

        attribute = 'SPINDENSITY_MATRIX'
        return np.asarray(self.maybe_fetch_dset(attribute))

    def supsym_irrep_indices(self):

        try:
            indices = self.maybe_fetch_dset('SUPSYM_IRREP_INDICES')
        except DataNotAvailable:
            indices = np.zeros(sum(self.n_bas**2), dtype=int)
        return indices

    def supsym_irrep_labels(self):

        try:
            data_bytes = self.maybe_fetch_dset('SUPSYM_IRREP_LABELS')
            labels = np.asarray(data_bytes, dtype='U')
        except DataNotAvailable:
            labels = np.array(['-']*sum(self.n_bas), dtype='U')
        return labels

    def write(self, wfn):
        self.h5f.attrs['NSYM'] = wfn.n_sym
        self.h5f.attrs['NBAS'] = np.array(wfn.n_bas)
        for kind in wfn.mo.keys():
            orbitals = wfn.mo[kind]
            field = self._get_mo_attribute('VECTORS', kind=kind)
            self.h5f[field] = np.ravel(orbitals.coefficients)
            field = self._get_mo_attribute('OCCUPATIONS', kind=kind)
            self.h5f[field] = np.ravel(orbitals.occupations)
            field = self._get_mo_attribute('ENERGIES', kind=kind)
            self.h5f[field] = np.ravel(orbitals.energies)
            field = self._get_mo_attribute('TYPEINDICES', kind=kind)
            self.h5f[field] = np.ravel(orbitals.types).astype('S')
            self.h5f['SUPSYM_IRREP_INDICES'] = np.ravel(orbitals.irreps)
