"""Shared Decisions answer fixture for placement semantic tests."""


class Answers:
    def __init__(self, **values):
        self.values = values
        self.calls = []

    async def __call__(self, requests, extra):
        self.calls.append((requests, extra))
        judgments = {}
        for request in requests:
            value = self.values.get(request.name, 'unknown')
            judgments[request.name] = ({'status': 'yes' if value is True else 'no' if value is False else 'unknown'}
                if not hasattr(request, 'choices') else
                {'status': 'unknown' if value == 'unknown' else
                 'proposed' if request.selection_mode == 'proposal' else 'known',
                 'value': value, 'confidence': .97})
        return {'judgments': judgments, 'evidence': {'request_hash': 'decision-' + str(len(self.calls))}}

