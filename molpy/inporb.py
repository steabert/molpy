from . import export
from .tools import *


@export
class MolcasINPORB():

    def __init__(self, filename, mode):
        """ initialize from a file object """
        self.f = open(filename, mode)

    def read(self):
        self.read_version()
        self.read_info()
        self.read_orb()
        self.read_occ()
        self.read_one()
        self.read_index()

    def write(self, wfn):
        self.write_version(wfn.molcas_version)
        self.write_info(wfn.nsym, wfn.nbas)
        self.write_orb(wfn.mo_vectors)
        self.write_occ(wfn.mo_occupations)
        self.write_one(wfn.mo_energies)
        self.write_index(wfn.mo_typeindices)

    # required by the MolcasWFN class
    def nsym(self):
        return self._nsym

    def nbas(self):
        return self._nbas

    def mo_vectors(self, nbas, uhf):
        data_flat = self._orb
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

    def mo_occupations(self, nbas, uhf):
        data_flat = self._occ
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

        data_flat = self._one
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

    def mo_typeindices(self, nbas, uhf):
        data_flat = self._index
        if data_flat is not None:
            typeindices = arr_to_lst(np.char.lower(data_flat), nbas)
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

    def supsym_irrep_indices(self, nbas, uhf):
        return (None, None)

    def read_version(self):
        line = seek_line(self.f, '#INPORB')
        self._version = line.split()[1]

    def read_info(self):

        seek_line(self.f, '#INFO')
        self._isuhf, self._nsym, self._wftype = (int(val) for val in self._next_noncomment().split())
        self._nbas = np.array(self._next_noncomment().split(), dtype=np.int)
        self._norb = np.array(self._next_noncomment().split(), dtype=np.int)
        print(self._nsym, self._nbas)

    def read_orb(self):

        seek_line(self.f, '#ORB')
        self._orb = np.empty(sum(self._nbas**2), dtype=np.float64)
        # choose block reader
        if self._version == '1.1':
            read_block = self._read_block_v11
        elif self._version == '2.0':
            read_block = self._read_block_v20
        # read the values
        sym_offset = 0
        for nb in self._nbas:
            if nb == 0:
                continue
            for offset in range(sym_offset, sym_offset + nb**2, nb):
                self._orb[offset:offset+nb] = read_block(nb)
            sym_offset += nb**2

    def read_occ(self):

        seek_line(self.f, '#OCC')
        self._occ = np.empty(sum(self._nbas), dtype=np.float64)
        # choose block reader
        if self._version == '1.1':
            blk_size = 4
            read_block = self._read_block_v11
        elif self._version == '2.0':
            blk_size = 10
            read_block = self._read_block_v20
        # read the values
        sym_offset = 0
        for nb in self._nbas:
            self._occ[sym_offset:sym_offset+nb] = read_block(nb, blk_size)
            sym_offset += nb

    def read_one(self):

        seek_line(self.f, '#ONE')
        self._one = np.empty(sum(self._nbas), dtype=np.float64)
        # choose block reader
        if self._version == '1.1':
            blk_size = 4
            read_block = self._read_block_v11
        elif self._version == '2.0':
            blk_size = 10
            read_block = self._read_block_v20
        # read the values
        sym_offset = 0
        for nb in self._nbas:
            self._one[sym_offset:sym_offset+nb] = read_block(nb, blk_size)
            sym_offset += nb

    def read_index(self):

        seek_line(self.f, '#INDEX')
        self._index = np.empty(sum(self._nbas), dtype='U1')
        blk_size = 10
        sym_offset = 0
        for nb in self._nbas:
            for offset in range(sym_offset, sym_offset + nb, blk_size):
                values = self._next_noncomment().split()[1].strip()
                size = min(blk_size, sym_offset + nb - offset)
                self._index[offset:offset+size] = np.array([values]).view('U1')
            sym_offset += nb

    def write_version(self, version):
        self.f.write('#INPORB {:s}\n'.format(version))

    def write_info(self, sym_size, basis_sizes):
        """ write info block """
        self.f.write('#INFO\n')
        self.f.write((3 * '{:8d}' + '\n').format(0,sym_size,0))
        self.f.write((sym_size * '{:8d}' + '\n').format(*basis_sizes))
        self.f.write((sym_size * '{:8d}' + '\n').format(*basis_sizes))

    def write_orb(self, mo_vectors):

        self.f.write('#ORB\n')
        for isym, coef in enumerate(mo_vectors):
            norb = coef.shape[0]
            for jorb in range(norb):
                self.f.write('* ORBITAL{:5d}{:5d}\n'.format(isym+1,jorb+1))
                for iorb_sta in range(0,norb,5):
                    iorb_end = min(iorb_sta+5,norb)
                    n = iorb_end - iorb_sta
                    self.f.write((n * ' {:21.14E}' + '\n').format(*coef[iorb_sta:iorb_end,jorb]))

    def write_occ(self, mo_occupations):

        self.f.write('#OCC\n* OCCUPATION NUMBERS\n')
        for occ in mo_occupations:
            self._write_blocked(occ, ' {:6.4f}', blocksize=10)

    def write_one(self, mo_energies):

        self.f.write('#ONE\n* ONE ELECTRON ENERGIES\n')
        for ene in mo_energies:
            self._write_blocked(ene, ' {:11.4f}', blocksize=10)

    def write_index(self, mo_typeindices):

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


#@export
#def open(filename, mode):
#    return MolcasINPORB(open(filename, mode))
