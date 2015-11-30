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
