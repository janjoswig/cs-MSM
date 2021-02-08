import pytest

from csmsm import examples


@pytest.fixture
def registered_dtrajs(request):
    """Return requested registered discrete trajectory by key"""
    key = request.node.funcargs.get("registered_key")

    return examples.registered[key]
