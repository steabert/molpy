from . import export
from .tools import *


@export
class MolcasINPORB():

    def __init__(self, filename, mode, version='2.0'):
        """ initialize from a file object """
        self.f = open(filename, mode)

        if mode.startswith('r'):
            line = seek_line(self.f, '#INPORB')
            self.version = line.split()[1]
            seek_line(self.f, '#INFO')
            self.unrestricted, self.n_sym, self.wfn_type = (int(val) for val in self._next_noncomment().split())
            self.n_bas = np.array(self._next_noncomment().split(), dtype=np.int)
            self.n_orb = np.array(self._next_noncomment().split(), dtype=np.int)
        elif mode.startswith('w'):
            self.version = version
        else:
            raise Exception('invalid mode string')

        if self.version == '2.0':
            self.read_block = self._read_block_v20
            self.occ_fmt = ' {:7.4f}'
            self.one_fmt = ' {:12.4e}'
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

    def read(self):
        self.read_info()
        self.read_orb()
        self.read_occ()
        self.read_one()
        self.read_index()

    def write(self, wfn):
        self.write_version(self.version)
        self.write_info(1, [wfn.basis_set.n_cgto])
        for kind in wfn.mo.keys():
            orbitals = wfn.mo[kind]
            self.write_orb([orbitals.coefficients], kind=kind)
            self.write_occ([orbitals.occupations], kind=kind)
            self.write_one([orbitals.energies], kind=kind)
            self.write_index([orbitals.types], kind=kind)

    def close(self):
        self.f.close()

    def read_orb(self, kind='restricted'):

        seek_line(self.f, self._format_header('ORB', kind=kind))
        coefficients = np.empty(sum(self.n_bas**2), dtype=np.float64)
        sym_offset = 0
        for nb in self.n_bas:
            if nb == 0:
                continue
            for offset in range(sym_offset, sym_offset + nb**2, nb):
                coefficients[offset:offset+nb] = self.read_block(nb)
            sym_offset += nb**2
        return arr_to_lst(coefficients, [(nb,nb) for nb in self.n_bas])

    def read_occ(self, kind='restricted'):

        seek_line(self.f, self._format_header('OCC', kind=kind))
        occupations = np.empty(sum(self.n_bas), dtype=np.float64)
        sym_offset = 0
        for nb in self.n_bas:
            occupations[sym_offset:sym_offset+nb] = self.read_block(nb, self.occ_blk_size)
            sym_offset += nb
        return arr_to_lst(occupations, self.n_bas)

    def read_one(self, kind='restricted'):

        seek_line(self.f, self._format_header('ONE', kind=kind))
        energies = np.empty(sum(self.n_bas), dtype=np.float64)
        sym_offset = 0
        for nb in self.n_bas:
            energies[sym_offset:sym_offset+nb] = self.read_block(nb, self.one_blk_size)
            sym_offset += nb
        return arr_to_lst(energies, self.n_bas)

    def read_index(self):

        seek_line(self.f, '#INDEX')
        typeindices = np.empty(sum(self.n_bas), dtype='U1')
        blk_size = 10
        sym_offset = 0
        for nb in self.n_bas:
            for offset in range(sym_offset, sym_offset + nb, blk_size):
                values = self._next_noncomment().split()[1].strip()
                size = min(blk_size, sym_offset + nb - offset)
                typeindices[offset:offset+size] = np.array([values]).view('U1')
            sym_offset += nb
        return arr_to_lst(typeindices, self.n_bas)

    def write_version(self, version):
        self.f.write('#INPORB {:s}\n'.format(version))

    def write_info(self, sym_size, basis_sizes):
        """ write info block """
        self.f.write('#INFO\n')
        self.f.write((3 * '{:8d}' + '\n').format(0,sym_size,0))
        self.f.write((sym_size * '{:8d}' + '\n').format(*basis_sizes))
        self.f.write((sym_size * '{:8d}' + '\n').format(*basis_sizes))

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
            line = self._next_noncomment()
            values = [line[sta:sta+18] for sta in range(0,72,18)]
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
                prefix = '{:1d} '.format(idx)
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


@export
class MolcasINPORB11(MolcasINPORB):
    def __init__(self, filename, mode):
        super().__init__(filename, mode, version='1.1')


@export
class MolcasINPORB20(MolcasINPORB):
    def __init__(self, filename, mode):
        super().__init__(filename, mode, version='2.0')
