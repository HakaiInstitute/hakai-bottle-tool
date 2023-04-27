from setuptools import find_packages, setup

setup(
    name="hakai_bottle_tool",
    version="1.0.1",
    packages=find_packages(),
    py_modules=["hakai_bottle_tool"],
    url="hakai.org",
    license="",
    author="Jessy Barrette",
    author_email="Jessy.Barrette@hakai.org",
    description="Method use to combine Hakai sample data to CTD profile data.",
    install_requires=["pandas", "numpy", "hakai_api", "plotly", "xarray", "NetCDF4"],
    include_package_data=True,
)
