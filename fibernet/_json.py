"""Private JSON object construction shared by native document readers."""


def unique_object_pairs(pairs):
    """Keep ordinary dictionary semantics, rejecting repeated decoded keys."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result
