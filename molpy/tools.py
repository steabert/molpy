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
    return np.array(sorted(np.arange(len(lst)), key=lst.__getitem__))


def seek_line(f, pattern):
    """ find the next line starting with a specific string """
    line = next(f)
    while not line.startswith(pattern):
        line = next(f)
    return line


def ordered_list(lst):
    """ create a dictionary representing an ordered list """
    return dict([(item, idx) for idx, item, in enumerate(lst)])
