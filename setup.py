from setuptools import setup

setup(
    name="hakai-bottle-tool",
    version="1.0",
    packages=["hakai_bottle_tools"],
    url="",
    license="",
    author="Jessy Barrette",
    author_email="Jessy.Barrette@hakai.org",
    description="Method use to combine Hakai sample data to CTD profile data.",
    install_requires=["pandas", "numpy", "hakai_api"],
)
