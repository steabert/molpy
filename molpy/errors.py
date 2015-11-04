from . import export


@export
class Error(Exception):
    """ Base class for errors """


@export
class UserError(Error):
    """ The user has made an error """
