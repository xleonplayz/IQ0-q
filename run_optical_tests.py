def run_tests():
    import pytest
    pytest.main(['tests/core/test_optical_processes.py', '-v'])

if __name__ == '__main__':
    run_tests()
