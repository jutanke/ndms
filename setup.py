from distutils.core import setup

setup(
    name="ndms",
    version="0.0.5",
    packages=[
        "ndms",
    ],
    setup_requires=["numpy", "numba"],
    install_requires=["numpy", "numba", "annoy"],
)
