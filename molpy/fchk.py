# fchk.py -- Guassian Formatted Checkpoint format
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

import copy
import numpy as np

from . import export
from .errors import DataNotAvailable


@export
class MolcasFCHK:

    max_angmom = 3

    def __init__(self, filename, mode):
        self.f = open(filename, mode)

    def write(self, wfn):
        if wfn.basis_set is None:
            raise DataNotAvailable('The fchk format needs basis set info.')
        self.write_header(
            'Molcas -> gaussian formatted checkpoint',
            'type',
            'method',
            'basis'
            )
        n_atoms, nuclear_charge = wfn.nuclear_info()
        assert not np.isnan(nuclear_charge)
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
        self.write_info(
            n_atoms,
            charge,
            spinmult,
            n_electrons,
            n_a,
            n_b,
            wfn.basis_set.n_cgto,
            )
        self.write_atom_info(
            wfn.basis_set.center_charges,
            wfn.basis_set.center_coordinates,
            )
        self.write_basisset(
            wfn.basis_set.primitive_tree,
            wfn.basis_set.center_coordinates,
            )
        if wfn.unrestricted:
            kinds = ['alpha', 'beta']
        else:
            kinds = ['restricted']
        for kind in kinds:
            orbitals = wfn.mo[kind].sort_basis(order='molden')
            orbitals = orbitals.limit_basis(limit=self.max_angmom)
            orbitals.sanitize()
            self.write_orbitals(
                orbitals,
                kind=kind,
                )

    def close(self):
        self.f.close()

    def write_scalar_int(self, name, value):
        self.f.write('{:40s}   {:1s}     {:12d}\n'.format(name, 'I', int(value)))

    def write_scalar_real(self, name, value):
        self.f.write('{:40s}   {:1s}     {:22.15e}\n'.format(name, 'I', value))

    def write_scalar_string(self, name, value):
        self.f.write('{:40s}   {:1s}     {:12s}\n'.format(name, 'C', value))

    def write_scalar_logical(self, name, value):
        self.f.write('{:40s}   {:1s}     {:1s}\n'.format(name, 'L', 'T' if value else 'F'))

    def write_array_int(self, name, array):
        n = array.size
        a = np.ravel(array, order='F')
        self.f.write('{:40s}   {:1s}   N={:12d}\n'.format(name, 'I', n))
        record_size = 6
        for offset in range(0, n, record_size):
            line = ''.join('{:12d}'.format(x) for x in a[offset:offset+record_size])
            print(line, file=self.f)

    def write_array_real(self, name, array):
        n = array.size
        a = np.ravel(array, order='F')
        self.f.write('{:40s}   {:1s}   N={:12d}\n'.format(name, 'R', n))
        record_size = 5
        for offset in range(0, n, record_size):
            line = ''.join('{:16.8e}'.format(x) for x in a[offset:offset+record_size])
            print(line, file=self.f)

    def write_array_string(self, name, array):
        n = array.size
        a = np.ravel(array, order='F')
        self.f.write('{:40s}   {:1s}   N={:12d}\n'.format(name, 'C', n))
        record_size = 5
        for offset in range(0, n, record_size):
            line = ''.join('{:12s}'.format(x) for x in a[offset:offset+record_size])
            print(line, file=self.f)

    def write_array_logical(self, name, array):
        n = array.size
        a = np.ravel(array, order='F')
        self.f.write('{:40s}   {:1s}   N={:12d}\n'.format(name, 'C', n))
        record_size = 72
        for offset in range(0, n, record_size):
            line = ''.join('{:1s}'.format(x) for x in a[offset:offset+record_size])
            print(line, file=self.f)

    def write_header(self, title, calctype, method, basis):
        self.f.write('{:72s}\n'.format(title))
        self.f.write('{:10s}{:30s}{:30s}\n'.format(calctype, method, basis))

    def write_info(self, natoms, charge, spin, nelec, nelec_a, nelec_b, nbas):
        self.write_scalar_int('Number of atoms', natoms)
        self.write_scalar_int('Charge', charge)
        self.write_scalar_int('Multiplicity', spin)
        self.write_scalar_int('Number of electrons', nelec)
        self.write_scalar_int('Number of alpha electrons', nelec_a)
        self.write_scalar_int('Number of beta electrons', nelec_b)
        self.write_scalar_int('Number of basis functions', nbas)

    def write_atom_info(self, charges, coords):
        self.write_array_int('Atomic numbers', charges.astype(np.int64))
        self.write_array_real('Nuclear charges', charges)
        self.write_array_real('Current cartesian coordinates', coords.T)

    def write_basisset(self, basisset, coords):
        contracted_shells = 0
        largest_contraction = 0
        highest_angmom = 0
        primitive_shells = 0
        for center in basisset:
            for angmom in center['angmoms']:
                if angmom['value'] > self.max_angmom:
                    continue
                highest_angmom = max(angmom['value'], highest_angmom)
                contracted_shells += len(angmom['shells'])
                for shell in angmom['shells']:
                    nprim = len(shell['exponents'])
                    primitive_shells += nprim
                    largest_contraction = max(nprim, largest_contraction)
        self.write_scalar_int('Number of contracted shells', contracted_shells)
        self.write_scalar_int('Number of primitive shells', primitive_shells)
        self.write_scalar_int('Highest angular momentum', highest_angmom)
        self.write_scalar_int('Largest degree of contraction', largest_contraction)
        shell_types = np.empty((contracted_shells,), order='F', dtype=np.int32)
        shell_to_atom_map = np.empty((contracted_shells,), order='F', dtype=np.int32)
        shell_coordinates = np.empty((3,contracted_shells), order='F')
        primitives_per_shell = np.empty((contracted_shells,), order='F', dtype=np.int32)
        exponents = np.empty((primitive_shells,), order='F')
        coefficients = np.empty((primitive_shells,), order='F')
        ishell = 0
        offset = 0
        for center in basisset:
            for angmom in center['angmoms']:
                if angmom['value'] > self.max_angmom:
                    continue
                highest_angmom = max(angmom['value'], highest_angmom)
                contracted_shells += len(angmom['shells'])
                for shell in angmom['shells']:
                    shell_types[ishell] = angmom['value'] * (-1)**(angmom['value']//2)
                    shell_to_atom_map[ishell] = center['id']
                    shell_coordinates[:,ishell] = coords[center['id']-1,:]
                    nprim = len(shell['exponents'])
                    primitives_per_shell[ishell] = nprim
                    exponents[offset:offset+nprim] = shell['exponents']
                    coefficients[offset:offset+nprim] = shell['coefficients']
                    offset += nprim
                    ishell += 1
        self.write_array_int('Shell types', shell_types)
        self.write_array_int('Number of primitives per shell', primitives_per_shell)
        self.write_array_int('Shell to atom map', shell_to_atom_map)
        self.write_array_real('Primitive exponents', exponents)
        self.write_array_real('Contraction coefficients', coefficients)
        self.write_array_real('Coordinates of each shell', shell_coordinates)

    def write_orbitals(self, orbitals, kind='restricted'):
        self.write_scalar_int('Number of basis functions', orbitals.n_bas)
        if kind == 'restricted' or kind == 'alpha':
            prefix = 'Alpha '
        elif kind == 'beta':
            prefix = 'Beta '
        self.write_array_real(prefix + 'Orbital Energies', orbitals.energies)
        self.write_array_real(prefix + 'MO coefficients', orbitals.coefficients)
