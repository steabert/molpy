# orbitals.py -- Molecular orbitals
#
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
# Modified by Marcus Johansson to show symmetry species.
#

import re
import numpy as np

from . import export
from .errors import Error, DataNotAvailable

try:
    import libmsym as msym
except ImportError:
    msym = None


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


@export
class OrbitalSet():
    """
    Represents a set of orbitals with a common basis set, and keeps track of
    their properties (energy, occupation, type, irrep).
    """
    def __init__(self, coefficients, ids=None, types=None,
                 irreps=None, energies=None, occupations=None,
                 basis_ids=None, basis_set=None):

        self.coefficients = np.asarray(coefficients)
        self.n_bas = coefficients.shape[0]
        self.n_orb = coefficients.shape[1]

        if irreps is None:
            self.irreps = np.zeros(self.n_bas)
            self.n_irreps = 1
        else:
            self.irreps = irreps
            self.n_irreps = len(np.unique(irreps))

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

        try:
            ids = self.basis_set.argsort_ids(self.basis_ids, order=order)
            return self.filter_basis(ids)
        except AttributeError:
            raise DataNotAvailable('A basis set is required to order basis functions')

    def limit_basis(self, limit=3):

        try:
            ids = self.basis_set.angmom_ids(self.basis_ids, limit=limit)
            return self.filter_basis(ids)
        except AttributeError:
            raise DataNotAvailable('A basis set is required to limit basis functions')

    def collapse(self, threshold=0.1):

        if threshold is None:
            return self

        rows = np.logical_or.reduce(abs(self.coefficients) > threshold, axis=1)
        return self.filter_basis(rows)

    def __str__(self):
        """
        returns the Orbital coefficients formatted as columns
        """

        prefix = '{:16s}'
        int_template = prefix + self.n_orb * '{:10d}'
        float_template = prefix + self.n_orb * '{:10.4f}'
        str_template = prefix + self.n_orb * '{:>10s}'

        lines = []

        line = int_template.format("ID", *self.ids)
        lines.append(line)

        line = str_template.format('', *(['------'] * self.n_orb))
        lines.append(line)

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

        lines.append('')

        return '\n'.join(lines)

    def show(self, cols=10, threshold=None):
        """
        prints the entire orbital set in blocks of cols orbitals
        """

        if self.n_orb == 0:
            print('no orbitals to show... perhaps you filtered too strictly?')

        for offset in range(0, self.n_orb, cols):
            orbitals = self[offset:offset+cols].collapse(threshold=threshold)
            print(orbitals)

    def show_by_irrep(self, cols=10, threshold=None):

        if self.n_irreps > 1:
            for irrep in np.unique(self.irreps):
                print('symmetry {:d}'.format(irrep+1))
                print()
                indices, = np.where(self.irreps == irrep)
                self[indices].sorted(reindex=True).show(cols=cols, threshold=threshold)
        else:
            self.show(cols=cols, threshold=threshold)

    def show_symmetry_species(self):
        if msym is None:
            raise ImportError('no libmsym installation found')
        bs = self.basis_set

        elements = [msym.Element(coordinates = Coord, charge = int(Charge))
                    for Coord, Charge in zip(bs.center_coordinates, bs.center_charges)]

        basis_functions = [msym.RealSphericalHarmonic(element = elements[element_id-1], n = n + l, l = l, m = m)
                           for [element_id, n, l, m] in bs.contracted_ids]

        with msym.Context(elements = elements, basis_functions = basis_functions) as ctx:
            point_group = ctx.find_symmetry()
            species_names = [s.name for s in ctx.character_table.symmetry_species]
            coefficients = np.asarray(self.coefficients)
            for offset in range(0,self.n_orb):
                mo = coefficients[:,offset]
                #mo /= np.linalg.norm(mo)
                species_components = ctx.symmetry_species_components(mo)
                print(offset,': ',' + '.join(['%f %s' % k for k in zip(species_components, species_names) if k[0] > 1.0e-6]))

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
        matching = [bool(re.search(regex, label, re.IGNORECASE)) for label in self.basis_set.labels]

        return self.filter_basis(np.asarray(matching))

    def sanitize(self):
        """
        Sanitize the orbital data, replacing NaN or missing values with safe
        placeholders.
        """
        for attribute in ['occupations', 'energies', 'coefficients']:
            array = getattr(self, attribute)
            selection = np.where(np.isnan(array))
            array[selection] = 0.0

        selection = np.where(self.types == '-')
        self.types[selection] = 's'
