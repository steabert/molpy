from . import export


@export
class Error(Exception):
    """ Base class for errors """


@export
class UserError(Error):
    """ The user has made an error """

@export
class InvalidRequest(Error):
    """ Wrong data was requested """

@export
class DataNotAvailable(Error):
    """ The data requested is not available """
