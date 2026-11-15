"""Available explicit graph routes; no model prompt assembly."""


def available_branch_choices(choices, branches):
    """An exhausted edge remains in the trace but is no longer a selectable operation."""
    available = {branch['outcome'] for branch in branches if branch['remaining_traversals'] > 0}
    return tuple(choice for choice in choices if choice == 'unknown' or choice in available)
