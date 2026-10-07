"""Explanatory drawings for decision checks 7--10; no optimized geometry."""
import numpy as np
from matplotlib.patches import Circle, Rectangle

from drawing import (View, setup, text, arrow, line, curve, arc, circle,
                     elevated_transfer, save_single, TEAL, ORANGE, PURPLE,
                     RED, GREEN, INK, MUTED)


def _transfer_parts(view, path):
    """Orange approach bends and purple high parts; every segment is drawn."""
    high = path[:, 2] > .94*path[:, 2].max()
    boundaries = [0, *(np.flatnonzero(high[1:] != high[:-1])+1), len(path)-1]
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        color = PURPLE if high[(first+last)//2] else ORANGE
        view.path(path[first:last+1], color, 3.0)


def _endpoints(view, first, last):
    for p, name, dx in ((first, 'A', -3), (last, 'B', 3)):
        q = view.p(p)
        view.ax.plot(*q, 'o', color=INK, ms=4, zorder=10)
        text(view.ax, q[0]+dx, q[1]-4, name, 14.5)


def draw7(ax):
    setup(ax, 7, 'Consolidate optional jumps',
          'Same endpoints and obstacle turns.\n'
          'Remove only optional landings.\n'
          'Keep every required winding interval.')
    for humps, box, y in ((3, (15, 55, 78, 27), 78),
                          (1, (15, 23, 78, 27), 46)):
        v = View(ax, box=box, bounds=(-36, 36, -15, 15, 0, 16))
        v.grid()
        for x in (-22, 0, 22):
            v.path([[x, -13, 0], [x, 13, 0]], TEAL, 3.2)
        p = elevated_transfer(-33, 33, height=13, humps=humps, n=601)
        _transfer_parts(v, p)
        _endpoints(v, p[0], p[-1])
        v.direction(p[300], [1, 0, 0], length=7, color=PURPLE)
        text(ax, 3, y, f'J = {humps}', 16, ha='left', weight='normal')
        if humps == 3:
            for x in (-11, 11):
                q = v.p([x, 0, 0])
                ax.plot(*q, 'o', color=ORANGE, ms=4)
    text(ax, 51, 86, 'Three separate rises and landings', 14.5)
    text(ax, 53, 54, 'One continuous raised span', 14.5)


def draw8(ax):
    setup(ax, 8, 'Align returns with B₀',
          'Straight span: no field along B₀.\n'
          'Both end bends remain in the field check.')
    v = View(ax, box=(6, 32, 88, 44), bounds=(-38, 38, -23, 23, 0, 18))
    v.grid()
    for x in (-16, 0, 16):
        v.path([[x, -17, 0], [x, 17, 0]], TEAL, 2.7)
    p = elevated_transfer(-33, 33, height=15, humps=1, n=601)
    # The helper's middle is exactly straight: colour every curved part orange.
    first = 90
    last = 510
    v.path(p[:first+1], ORANGE, 3.2)
    v.path(p[first:last+1], PURPLE, 3.2)
    v.path(p[last:], ORANGE, 3.2)
    v.direction(p[300], [1, 0, 0], length=10, color=PURPLE)
    v.b0(p=(-29, -21, 0), length=24)
    text(ax, 50, 82, 'Straight span only:  ' + r'$B_\parallel=0$', 15.5)
    high = v.p(p[330])
    arrow(ax, (56, 76), high+(0, 2), PURPLE, 1.3, 10, '-')
    text(ax, 50, 28, 'Orange: both smooth end bends', 14.5, ORANGE)


def _capsule(ax, x0, x1, y, width, color):
    ax.add_patch(Rectangle((x0, y-width/2), x1-x0, width,
                           facecolor=color, edgecolor='none', alpha=.88))
    for x in (x0, x1):
        ax.add_patch(Circle((x, y), width/2, facecolor=color,
                            edgecolor='none', alpha=.88))
    line(ax, [[x0, y], [x1, y]], '#ffffff', .7, '--', .85)


def draw9(ax):
    setup(ax, 9, 'Reject infeasible routes',
          'Check width, gap, bends and current.\n'
          'Use the stated manufacturing limits.')
    # Two finite-width conductors: the relevant distance is edge-to-edge.
    _capsule(ax, 9, 37, 79, 5, TEAL)
    _capsule(ax, 9, 37, 71, 5, RED)
    arrow(ax, (43, 73.5), (43, 76.5), INK, 1.0, 8, '<->')
    text(ax, 48, 75, 'g', 14.5)
    text(ax, 56, 76, 'Edge clearance\n' + r'$g < g_{\min}$', 15, RED, ha='left')

    # A finite-width bend, with a symbolic centreline radius R.
    theta = np.linspace(-np.pi/2, 0, 50)
    bend = np.column_stack((25+5*np.cos(theta), 54+5*np.sin(theta)))
    p = np.vstack(([[9, 49]], bend, [[30, 62]]))
    line(ax, p, RED, 12)
    line(ax, p, '#ffffff', .7, '--')
    ax.plot(25, 54, '+', color=INK, ms=5)
    arrow(ax, (25, 54), (30, 54), INK, 1.0, 8, '<->')
    text(ax, 43, 54, 'R', 14.5)
    text(ax, 56, 55, 'Bend radius\n' + r'$R < R_{\min}$', 15, RED, ha='left')

    # Geometric continuity alone is insufficient: arrows must be compatible.
    line(ax, [[8, 31], [22, 31]], TEAL, 4)
    line(ax, [[22, 31], [34, 31]], ORANGE, 4)
    line(ax, [[34, 31], [47, 31]], TEAL, 4)
    arrow(ax, (12, 31), (20, 31), INK, 1.7, 13)
    arrow(ax, (44, 31), (36, 31), INK, 1.7, 13)
    ax.plot(29, 31, 'x', color=RED, ms=12, mew=2.3, zorder=12)
    text(ax, 56, 31, 'Required current\nmust be continuous', 14.5, RED, ha='left')


def _roi(view):
    view.path(circle(5.2, center=(0, 0, -7)), GREEN, 1.6, '--')
    p = view.p([0, 0, -7])
    text(view.ax, p[0], p[1], 'ROI', 14.5, GREEN)


def _directed_join(first, second):
    ta = first[-1]-first[-2]
    ta /= np.linalg.norm(ta)
    tb = second[1]-second[0]
    tb /= np.linalg.norm(tb)
    distance = .3*np.linalg.norm(second[0]-first[-1])
    return curve([first[-1], first[-1]+distance*ta,
                  second[0]-distance*tb, second[0]])


def draw10(ax):
    setup(ax, 10, 'Evaluate the full field change',
          'Include removed arcs, joins and leads.\n'
          'Same current and target samples.\n'
          'No numerical field result is implied.')
    bounds = (-21, 29, -21, 23, -10, 16)
    left = View(ax, box=(2, 39, 45, 42), bounds=bounds)
    right = View(ax, box=(52, 39, 45, 42), bounds=bounds)
    text(ax, 25, 85, 'Original loops', 15.5)
    text(ax, 75, 85, 'Complete route', 15.5)
    left.grid()
    right.grid()
    _roi(left)
    _roi(right)
    retained = [arc(r, gap=.30) for r in (16, 10)]
    for radius, q in zip((16, 10), retained):
        left.path(circle(radius), TEAL, 2.8)
        right.path(q, TEAL, 2.8)
        left.direction([-radius, 0, 0], [0, -1, 0], length=7)
        right.direction([-radius, 0, 0], [0, -1, 0], length=7)
        theta = np.linspace(-.30, .30, 25)
        removed = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), np.zeros(len(theta))))
        right.path(removed, RED, 2.0, '--')
    right.path(_directed_join(*retained), ORANGE, 2.8)
    # This entire return closes the routed conductor; there are no free endpoints.
    finish, start = retained[-1][-1], retained[0][0]
    high1, high2 = np.array([25., -10., 13.]), np.array([25., 12., 13.])
    rise = curve([finish, finish+[3, 0, 5], high1+[0, -5, 0], high1])
    landing = curve([high2, high2+[0, 5, 0], start+[3, 3, 5], start])
    right.path(rise, ORANGE, 2.8)
    right.path([high1, high2], PURPLE, 2.8)
    right.path(landing, ORANGE, 2.8)
    right.direction((high1+high2)/2, [0, 1, 0], length=7, color=PURPLE)
    text(ax, 77, 77, 'all leads', 14.5, PURPLE)
    text(ax, 50, 35, 'Dashed red: removed contour arcs', 14.5, RED)
    text(ax, 50, 28.5, 'Same prescribed I; same ROI', 14.5)
    text(ax, 50, 21, r'$\Delta B = B_{\mathrm{complete}}-B_{\mathrm{original}}$', 17)


DRAWERS = {7: draw7, 8: draw8, 9: draw9, 10: draw10}


if __name__ == '__main__':
    for number, draw in DRAWERS.items():
        save_single(number, draw)
