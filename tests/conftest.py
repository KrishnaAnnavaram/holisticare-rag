from dataclasses import replace

import pytest

from holisticare_rag import synthetic
from holisticare_rag.config import Settings
from holisticare_rag.service import Assistant


@pytest.fixture()
def kb(tmp_path):
    synthetic.write(tmp_path)
    return tmp_path


@pytest.fixture()
def settings(kb):
    return replace(Settings(), docs_dir=kb / "documents", index_dir=kb / "index")


@pytest.fixture()
def assistant(settings):
    return Assistant.from_settings(settings)[0]
