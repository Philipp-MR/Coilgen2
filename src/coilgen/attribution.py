"""Software credits, separate from coil design authorship and numerical inputs."""

SOFTWARE_NAME = "CoilGen2"
AUTHOR_NAME = "Philipp Amrein"
AUTHOR_ROLE = "MRI hardware engineer"
AUTHOR_CREDIT = f"Created and maintained by {AUTHOR_NAME}"
PROJECT_URL = "https://github.com/Philipp-MR/Coilgen2"
COILGEN_DOI = "10.1002/mrm.29294"
COILGEN_PUBLICATION_URL = f"https://doi.org/{COILGEN_DOI}"
COILGEN_CITATION = (
    "Amrein P, Jia F, Zaitsev M, Littin S. CoilGen: Open-source MR coil layout generator. "
    "Magnetic Resonance in Medicine. 2022;88(3):1465–1479."
)


def software_attribution() -> dict[str, str]:
    """Return detached software provenance for newly generated artifacts."""
    return {
        "name": SOFTWARE_NAME,
        "creator": AUTHOR_NAME,
        "maintainer": AUTHOR_NAME,
        "creator_role": AUTHOR_ROLE,
        "repository": PROJECT_URL,
        "original_coilgen_doi": COILGEN_DOI,
    }
