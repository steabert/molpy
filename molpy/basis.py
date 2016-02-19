# basis.py - Basis set
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

import numpy as np

from . import export
from .errors import Error, DataNotAvailable

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

        self.cgto_molcas_rank = np.argsort(self.argsort(self._idtuples_ladder_order))
        self.cgto_molden_rank = np.argsort(self.argsort(self._idtuples_updown_order))

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
                if m == 0:
                    m_lbl = '0'
                else:
                    m_lbl = '{:+d}'.format(m)[::-1]
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

    def argsort_ids(self, ids=None, order=None):
        """
        Reorder the supplied ids of the contracted functions by either
        Molcas or Molden/Gaussian ranking and return an array of indices.
        """
        if ids is None:
            ids = np.arange(self.n_cgto)
        if order is None:
            return np.arange(len(ids))

        if order == 'molcas':
            rank = self.cgto_molcas_rank[ids]
        elif order == 'molden':
            rank = self.cgto_molden_rank[ids]
        else:
            raise Error('invalid order parameter')
        return np.argsort(rank)

    def angmom_ids(self, ids=None, limit=3):
        """
        Limit the basis ids by angular momentum. The default is to limit up
        to f functions (angular momentum 3).
        """
        if ids is None:
            ids = np.arange(self.n_cgto)

        cgto_angmom_ids = self.contracted_ids[ids,2]
        selection, = np.where(cgto_angmom_ids <= limit)
        return selection

    @staticmethod
    def argsort(lst, rank=None):
        """ sort indices of a list """
        return np.array(sorted(np.arange(len(lst)), key=lst.__getitem__))
