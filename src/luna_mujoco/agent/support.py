"""One raw-image support choice, only when measured support is unresolved."""
from ..core.decisions import ChoiceRequest


def support_question(subject, candidates, proposed_support=None):
    return ChoiceRequest(name='current_support', choices=(*candidates, 'none_of_listed', 'unknown'),
        descriptions={**{identifier: 'Support ID' + identifier + '. Compare its labelled CURRENT surface '
            'and dated reference evidence.' for identifier in candidates},
            'none_of_listed': 'A clearly visible support outside the offered list.',
            'unknown': 'The current support cannot be determined from the supplied evidence.'},
        instructions='Which support is the object ID' + subject['id'] + ' currently resting on or inside? '
        'Use CURRENT scoped RGB with labelled subject/support contours and boxes when supplied. '
        'Dated reference annotations identify earlier observations. A container and the table beneath '
        'it are different supports: an object among bin contents is inside that bin. '
        'Choose none_of_listed for an evident different support, unknown when unclear.')
