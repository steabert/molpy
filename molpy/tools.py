import sys
import numpy as np
import re


def info(*args):
    print(*args)


def warn(*args):
    print('Warning:', *args)


def error(*args):
    print('Error:', *args)
    sys.exit(1)


def print_line(name, lst):
    """ write a string + list of objects on one line """
    prefix = '{:16s}'
    n = len(lst)
    if (isinstance(lst[0], int)):
        body = " {:10d}"
    elif (isinstance(lst[0], np.int64)):
        body = " {:10d}"
    elif (isinstance(lst[0], np.float64)):
        body = " {:10.4f}"
    elif (isinstance(lst[0], str)):
        body = " {:>10s}"
    else:
        argtype = str(type(lst[0]))
        raise TypeError('cannot handle type {:s}'.format(argtype))
    fmt = prefix + n * body
    print(fmt.format(name, *lst))


def print_header(*args, width=128):
    delim = '*'
    starline = delim * width
    starskip = delim + ' ' * (width - 2) + delim
    print(starline)
    print(starskip)
    for arg in args:
        print(delim + arg.center(width - 2, ' ') + delim)
    print(starskip)
    print(starline)
    print()


def offsets(blk_sizes):
    """ compute the offsets of an array of block sizes """
    arr = np.array([np.prod(blk_size) for blk_size in blk_sizes])
    offsets = np.insert(np.cumsum(arr[:-1]), 0, 0)
    return offsets


def maybe_get(obj, attr, *args):
    """ call an attribute and return None when it doesn't exist """
    fun = getattr(obj, attr, None)
    if callable(fun):
        return fun(*args)
    else:
        warn("object {:s} has no attribute {:s}".format(str(obj), attr))
        return None


def arr_to_lst(arr, blk_sizes):
    """ convert a flat array to a list of (multi-dimensional) arrays """
    lst = []
    for blk_size, offset in zip(blk_sizes, offsets(blk_sizes)):
        arr_slice = arr[offset:offset+np.prod(blk_size)]
        lst.append(arr_slice.reshape(blk_size, order='F'))
    return lst


def lst_to_arr(lst):
    """ convert a list of (multi-dimensional) arrays to a single flat array """
    blk_sizes = [item.size for item in lst]
    arr = np.empty((sum(blk_sizes)), dtype=lst[0].dtype)
    for item, blk_size, offset in zip(lst, blk_sizes, offsets(blk_sizes)):
        arr[offset:offset+blk_size] = item.flatten('F')
    return arr


def argsort(lst, rank=None):
    """ sort indices of a list """
    if rank is None:
        def sortkey(x):
            return x[1]
    else:
        def sortkey(x):
            return rank(x[1])
    return np.array([i for i, elem in sorted(enumerate(lst), key=sortkey)])


def seek_line(f, pattern):
    """ find the next line starting with a specific string """
    line = next(f)
    while not line.startswith(pattern):
        line = next(f)
    return line


def ordered_list(lst):
    """ create a dictionary representing an ordered list """
    return dict([(item, idx) for idx, item, in enumerate(lst)])

angmom_name = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n']
l_index = ordered_list(angmom_name)


def filter_ao_labels(ao_labels, pattern=None, mx_angmom=None):
    if pattern is not None:
        label_match = np.vectorize(lambda lbl: bool(re.search(pattern, lbl)))
        indices, = np.where(label_match(ao_labels))
    elif mx_angmom is not None:
        label_match = np.vectorize(lambda lbl: l_index[lbl[7]])
        indices, = np.where(label_match(ao_labels) <= mx_angmom)
    else:
        indices = np.arange(len(ao_labels))
    return indices


def filter_mo_typeindices(mo_typeindices, typeids):
    if typeids is not None:
        pattern = '[' + ''.join(typeids) + ']'
        label_match = np.vectorize(lambda lbl: bool(re.match(pattern, lbl)))
        indices, = np.where(label_match(mo_typeindices))
    else:
        indices = np.arange(len(mo_typeindices))
    return indices

lenin = 6
lenin4 = lenin + 4


def rank_ao_tuple_molcas(ao_tuple):
    """
    rank a basis tuple according to Molcas ordering

    angmom components are ranked as (...,-2,-1,0,+1,+2,...)
    """
    center, n, l, m, = ao_tuple
    if l == 1 and m == 1:
        m = -2
    return (center, l, m, n)


def rank_ao_tuple_molden(ao_tuple):
    """
    rank a basis tuple according to Molden/Gaussian ordering

    angmom components are ranked as (0,1+,1-, 2+,2-,...)
    """
    center, n, l, m, = ao_tuple
    if l == 1 and m == 0:
        m = 2
    if m < 0:
        m = -m + 0.5
    return (center, l, n, m)


def format_ao_tuple(ao_tuple, center_labels):
    """ convert a basis function ID into a human-readable label """
    c, n, l, m, = ao_tuple
    center_lbl = center_labels[c-1]
    n_lbl = str(n+l) if n+l < 10 else '*'
    l_lbl = angmom_name[l]
    if l == 0:
        m_lbl = ''
    elif l == 1:
        m_lbl = ['y','z','x'][m+1]
    else:
        m_lbl = str(m)[::-1]
    return '{:6s}{:1s}{:1s}{:2s}'.format(center_lbl, n_lbl, l_lbl, m_lbl)


def safe_select(variable, fallback):
    """ return variable or fall back to value """
    if variable is not None:
        return variable
    else:
        return fallback


def scalify(lst):
    if lst is not None:
        return lst if len(lst) != 1 else lst[0]
    else:
        return None
