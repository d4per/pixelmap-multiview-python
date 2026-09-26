"""The exceptions this package raises.

Every one says which stage of the pipeline it came from and why, as attributes, so that a
caller can act on a failure without parsing its message. The message itself says what to
change about the photos.

The classes live here rather than in the native module because the Rust side constructs
them by importing this module, and because two base classes is a plain class statement in
Python.
"""

__all__ = [
    "PixelmapMultiviewError",
    "InvalidInputError",
    "ReconstructionError",
    "CancelledError",
]


class PixelmapMultiviewError(Exception):
    """Base class for every error raised by :mod:`pixelmap_multiview`.

    Attributes:
        stage: The stage that failed, one of the names in :data:`STAGES`, such as
            ``"registration"``.
        reason: Which failure it was, in snake_case, such as ``"too_few_tracks"`` or
            ``"bundle_adjustment"``. Stable across releases for the failures that exist
            today; match on this rather than on the message.

    Any further keyword arguments are the failure's own details, set as attributes of the
    same name — ``found`` and ``minimum`` for too few photos, ``median_px`` for a bundle
    adjustment that did not converge, and so on. :attr:`details` holds them all.
    """

    def __init__(self, message, stage=None, reason=None, **details):
        super().__init__(message)
        self.stage = stage
        self.reason = reason
        self.details = details
        for name, value in details.items():
            setattr(self, name, value)


class InvalidInputError(PixelmapMultiviewError, ValueError):
    """The photos or settings could not be used at all.

    Too few photos, photos of different sizes, a photo smaller than
    :data:`MIN_DIMENSION`, or an impossible focal length. Also a ``ValueError``, since
    that is what it means.
    """


class ReconstructionError(PixelmapMultiviewError):
    """The photos were valid, but no trustworthy model could be built from them.

    The pipeline fails loudly rather than returning a plausible-looking wrong model. The
    message says what to do differently: take the photos closer together, step sideways
    rather than turning on the spot, include things at different distances.
    """


class CancelledError(PixelmapMultiviewError):
    """The run was stopped by :meth:`Job.cancel`. :attr:`stage` says where."""
