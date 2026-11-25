"""Small physical PCB fixtures independent of saved-run packaging."""

from types import SimpleNamespace
import numpy as np
from coilgen.config import CoilGenConfig
from coilgen.models import CoilPart, Mesh, TargetField, WirePath
from coilgen.application.pcb import PcbSession


def plane_part(points=None):
    mesh = Mesh(np.array([[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0]]), np.array([[0, 1, 2], [0, 2, 3]]))
    points = np.array([[0.02, 0.03, 0], [0.08, 0.03, 0]]) if points is None else np.array(points, float)
    return CoilPart(mesh, wire_path=WirePath(points), contour_step=1.0)


def pcb_session():
    part = plane_part()
    coords = np.array([[0.05, 0.05, 0.02], [0.04, 0.05, 0.03]])
    target = TargetField(coords, np.zeros_like(coords), np.ones(2), np.zeros(2, bool), np.zeros(2, int))
    return PcbSession([part], CoilGenConfig(conductor_cross_section_width=0.001), target=target)


class State(dict):
    __getattr__ = dict.get

    def __setattr__(self, k, v):
        self[k] = v

    def flush(self):
        pass


def browser_pcb(tmp_path):
    from coilgen.web.pcb_layout import BrowserPcbLayout

    state = State()
    browser = SimpleNamespace(server=SimpleNamespace(state=state), exports=tmp_path, record=None)
    adapter = BrowserPcbLayout(browser)
    adapter.session = pcb_session()
    adapter.refresh()
    return adapter
