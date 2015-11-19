from . import export



class Error(Exception):
    """ Base class for errors """



class UserError(Error):
    """ The user has made an error """


class InvalidRequest(Error):
    """ Wrong data was requested """


class DataNotAvailable(Error):
    """ The data requested is not available """
