# errors.py -- Error handling
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

from . import export


@export
class Error(Exception):
    """ Base class for errors """
    prefix = 'Error:'
    def __str__(self):
        return ' '.join((self.prefix, super().__str__()))


@export
class UserError(Error):
    """ The user has made an error """


@export
class InvalidRequest(Error):
    """ Wrong data was requested """
    prefix = 'Error: invalid request:'


@export
class DataNotAvailable(Error):
    """ The data requested is not available """
    prefix = 'Error: missing data:'
