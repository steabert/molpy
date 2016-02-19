# inporb.py -- Molcas orbital format
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

from copy import deepcopy
import numpy as np

from . import export
from .errors import InvalidRequest


@export
class MolcasINPORB():
    """
    Handle reading and writing of Molcas INPORB files. These are the default
    formatted orbital files used by any Molcas program modules.
    """
    def __init__(self, filename, mode, version='2.0'):
        """
        initialize the INPORB file and prepare for reading or writing
        """
        if mode.startswith('r'):
            try:
                with open(filename, 'r') as f:
                    firstline = next(f)
                if not firstline.startswith('#INPORB'):
                    raise InvalidRequest
            except:
                raise InvalidRequest

        self.f = open(filename, mode)

        if mode.startswith('r'):
            line = self.seek_line('#INPORB')
            self.version = line.split()[1]
            self.seek_line('#INFO')
            uhf, self.n_sym, self.wfn_type = (int(val) for val in self._next_noncomment().split())
            if uhf == 1:
                self.unrestricted = True
            else:
                self.unrestricted = False
            self.n_bas = np.array(self._next_noncomment().split(), dtype=np.int)
            self.n_orb = np.array(self._next_noncomment().split(), dtype=np.int)
        elif mode.startswith('w'):
            self.version = version
        else:
            raise Exception('invalid mode string')

        if self.version == '2.0':
            self.read_block = self._read_block_v20
            self.occ_fmt = ' {:7.4f}'
            self.one_fmt = ' {:11.4e}'
            self.orb_fmt = ' {:21.14e}'
            self.occ_blk_size = 10
            self.one_blk_size = 10
            self.orb_blk_size = 5
        elif self.version == '1.1':
            self.read_block = self._read_block_v11
            self.occ_fmt = '{:18.11e}'
            self.one_fmt = '{:18.11e}'
            self.orb_fmt = '{:18.11e}'
            self.occ_blk_size = 4
            self.one_blk_size = 4
            self.orb_blk_size = 4
        else:
            raise Exception('invalid version number')

    def write(self, wfn):
        wfn = deepcopy(wfn)
        self.write_version(self.version)
        if wfn.unrestricted:
            uhf = 1
            kinds = ['alpha', 'beta']
        else:
            uhf = 0
            kinds = ['restricted']
        self.write_info(uhf, wfn.n_sym, wfn.n_bas)
        orbs = {}
        for kind in kinds:
            wfn.mo[kind].sanitize()
            orbs[kind] = wfn.symmetry_blocked_orbitals(kind=kind)
        for kind in kinds:
            self.write_orb((orb.coefficients for orb in orbs[kind]), kind=kind)
        for kind in kinds:
            self.write_occ((orb.occupations for orb in orbs[kind]), kind=kind)
        for kind in kinds:
            self.write_one((orb.energies for orb in orbs[kind]), kind=kind)
        for kind in kinds:
            self.write_index((orb.types for orb in orbs[kind]), kind=kind)

    def close(self):
        self.f.close()

    def rewind(self):
        self.f.seek(0)

    def read_orb(self, kind='restricted'):

        self.seek_line(self._format_header('ORB', kind=kind))
        coefficients = np.empty(sum(self.n_bas**2), dtype=np.float64)
        sym_offset = 0
        for nb in self.n_bas:
            if nb == 0:
                continue
            for offset in range(sym_offset, sym_offset + nb**2, nb):
                coefficients[offset:offset+nb] = self.read_block(nb)
            sym_offset += nb**2
        return coefficients

    def read_occ(self, kind='restricted'):

        self.seek_line(self._format_header('OCC', kind=kind))
        occupations = np.empty(sum(self.n_bas), dtype=np.float64)
        sym_offset = 0
        for nb in self.n_bas:
            occupations[sym_offset:sym_offset+nb] = self.read_block(nb, self.occ_blk_size)
            sym_offset += nb
        return occupations

    def read_one(self, kind='restricted'):

        self.seek_line(self._format_header('ONE', kind=kind))
        energies = np.empty(sum(self.n_bas), dtype=np.float64)
        sym_offset = 0
        for nb in self.n_bas:
            energies[sym_offset:sym_offset+nb] = self.read_block(nb, self.one_blk_size)
            sym_offset += nb
        return energies

    def read_index(self):

        self.seek_line('#INDEX')
        typeindices = np.empty(sum(self.n_bas), dtype='U1')
        blk_size = 10
        sym_offset = 0
        for nb in self.n_bas:
            for offset in range(sym_offset, sym_offset + nb, blk_size):
                values = self._next_noncomment().split()[1].strip()
                size = min(blk_size, sym_offset + nb - offset)
                typeindices[offset:offset+size] = np.array([values]).view('U1')
            sym_offset += nb
        return typeindices

    def write_version(self, version):
        self.f.write('#INPORB {:s}\n'.format(version))

    def write_info(self, uhf, n_sym, n_bas, title=''):
        """ write info block """
        self.f.write('#INFO\n')
        self.f.write(''.join(['*' + title + '\n']))
        self.f.write((3 * '{:8d}' + '\n').format(uhf,n_sym,0))
        self.f.write((n_sym * '{:8d}' + '\n').format(*n_bas))
        self.f.write((n_sym * '{:8d}' + '\n').format(*n_bas))

    def write_orb(self, mo_vectors, kind='restricted'):

        self.f.write(self._format_header('ORB', kind=kind))
        for isym, coef in enumerate(mo_vectors):
            norb = coef.shape[0]
            for jorb in range(norb):
                self.f.write('* ORBITAL{:5d}{:5d}\n'.format(isym+1,jorb+1))
                self._write_blocked(np.ravel(coef[:,jorb]), self.orb_fmt, blocksize=self.orb_blk_size)

    def write_occ(self, mo_occupations, kind='restricted'):

        self.f.write(self._format_header('OCC', kind=kind))
        self.f.write('* OCCUPATION NUMBERS\n')
        for occ in mo_occupations:
            self._write_blocked(occ, self.occ_fmt, blocksize=self.occ_blk_size)

    def write_one(self, mo_energies, kind='restricted'):

        self.f.write(self._format_header('ONE', kind=kind))
        self.f.write('* ONE ELECTRON ENERGIES\n')
        for ene in mo_energies:
            self._write_blocked(ene, self.one_fmt, blocksize=self.one_blk_size)

    def write_index(self, mo_typeindices, kind='restricted'):

        self.f.write('#INDEX\n')
        for idx in mo_typeindices:
            self.f.write('* 1234567890\n')
            self._write_blocked(idx, '{:1s}', blocksize=10, enum=True)

    # internal use

    def _next_noncomment(self):
        line = next(self.f)
        while line.startswith('*'):
            line = next(self.f)
        return line

    def _read_block_v11(self, size, blk_size=4):
        """
        read a block of 'size' values from an INPORB 1.1 file
        """
        arr = np.empty(size)
        for offset in range(0, size, blk_size):
            line = self._next_noncomment().rstrip()
            values = [line[sta:sta+18] for sta in range(0,len(line),18)]
            arr[offset:offset+blk_size] = np.array(values, dtype=np.float64)
        return arr

    def _read_block_v20(self, size, blk_size=5):
        """
        read a block of 'size' values from an INPORB 2.0 file
        """
        arr = np.empty(size)
        for offset in range(0, size, blk_size):
            values = self._next_noncomment().split()
            arr[offset:offset+blk_size] = np.array(values, dtype=np.float64)
        return arr

    def _write_blocked(self, arr, fmt, blocksize=5, enum=False):
        """
        write an array to file with a fixed number of elements per line
        """
        if enum:
            for idx, offset in enumerate(range(0, len(arr), blocksize)):
                prefix = '{:1d} '.format(idx % 10)
                line = ''.join(fmt.format(i) for i in arr[offset:offset+blocksize])
                self.f.write(prefix + line + '\n')
        else:
            for offset in range(0, len(arr), blocksize):
                line = ''.join(fmt.format(i) for i in arr[offset:offset+blocksize])
                self.f.write(line + '\n')

    @staticmethod
    def _format_header(header, kind='restricted'):
        if kind == 'beta':
            return '#U' + header + '\n'
        else:
            return '#' + header + '\n'

    def seek_line(self, pattern):
        """ find the next line starting with a specific string """
        line = next(self.f)
        while not line.startswith(pattern):
            line = next(self.f)
        return line


@export
class MolcasINPORB11(MolcasINPORB):
    def __init__(self, filename, mode):
        super().__init__(filename, mode, version='1.1')


@export
class MolcasINPORB20(MolcasINPORB):
    def __init__(self, filename, mode):
        super().__init__(filename, mode, version='2.0')
