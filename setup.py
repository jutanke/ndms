from distutils.core import setup

setup(
    name="ndms",
    version="0.0.3",
    packages=[
        "ndms",
    ],
    setup_requires=["numpy"],
    install_requires=["numpy", "annoy"],
)
