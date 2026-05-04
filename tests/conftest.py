"""Root conftest — register custom pytest marks."""


def pytest_configure(config):
    config.addinivalue_line('markers', 'invariant: boundary condition invariant test')
    config.addinivalue_line('markers', 'contract: slot interface contract test')
    config.addinivalue_line('markers', 'integration: touches real external services')
    config.addinivalue_line('markers', 'slow: single test > 10s')
