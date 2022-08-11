from distutils.core import setup

setup(
    name="ndms",
    version="0.1.3",
    packages=[
        "ndms",
    ],
    setup_requires=["numpy", "numba"],
    install_requires=["numpy", "numba", "annoy"],
)
