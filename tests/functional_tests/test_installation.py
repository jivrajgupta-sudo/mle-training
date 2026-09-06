import housing_project


def test_package_imports() -> None:
    assert housing_project.__version__ == "0.3.0"
