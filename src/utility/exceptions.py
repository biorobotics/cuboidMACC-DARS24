class TimeoutError(Exception):
    pass
class UnhandledConflictError(Exception):
    """
    Raised if we find a conflict that CBS doesn't know how to handle
    """
    pass
class InfeasibleAllocationError(Exception):
    """
    Raised if the task to robot allocation failed
    """
    pass