"""Decision diagrams for bounded search and lexicographic route selection."""
from drawing import (
    INK, MUTED, TEAL, ORANGE, PURPLE, GREEN, GRID,
    setup, text, arrow, box, line, save_single,
)


def draw11(ax):
    setup(
        ax, 11, 'Exhaust the jump budget',
        'Stopped early: not found / unresolved.\nThis is not an infeasibility proof.',
    )
    text(ax, 50, 88, 'Within declared candidate family', size=14.5,
         color=MUTED)

    box(ax, 7, 73, 43, 10, 'Fix jump count J', PURPLE)
    arrow(ax, (28.5, 73), (28.5, 68))
    box(ax, 7, 57, 43, 11, 'Try cuts / order /\nports', TEAL,
        size=14.5)
    arrow(ax, (28.5, 57), (28.5, 51))
    box(ax, 7, 40, 43, 11, 'Physical +\nfield checks', INK,
        size=14.5)

    arrow(ax, (50, 45.5), (61, 45.5), GREEN)
    box(ax, 61, 40, 35, 11, 'Pass: save route', GREEN, size=14.5)
    # A passing route remains a candidate while alternatives are untested.
    line(ax, [(78.5, 51), (78.5, 62.5)], GREEN, 1.8)
    arrow(ax, (78.5, 62.5), (50, 62.5), GREEN)
    text(ax, 75, 73, 'Untested remain:\nkeep searching', size=14.5,
         color=GREEN)

    # Rejecting one route is not exhaustion of the candidate family.
    line(ax, [(7, 45.5), (2.5, 45.5), (2.5, 62.5)], ORANGE, 1.8)
    arrow(ax, (2.5, 62.5), (7, 62.5), ORANGE)
    text(ax, 10, 53.7, 'Fail', size=14.5, color=ORANGE)

    line(ax, [(4, 36), (96, 36)], GRID, 1.3)
    text(ax, 50, 32.5, 'Only after exhausting this family', size=14.5,
         color=MUTED)
    arrow(ax, (26, 29.5), (26, 27), ORANGE)
    arrow(ax, (76, 29.5), (76, 27), GREEN)
    box(ax, 4, 17, 43, 10, 'None feasible:\ntry J + 1', ORANGE,
        size=14.5)
    box(ax, 55, 17, 41, 10, 'Feasible saved:\nchoose at this J', GREEN,
        size=14.5)


def draw12(ax):
    setup(
        ax, 12, 'Choose and recheck',
        'Symbolic errors: A or B may win.\nE = longitudinal field-error RMS.',
    )
    text(ax, 50, 86, 'Hypothetical feasible routes; returns included',
         size=14.5, color=MUTED)

    # Two equal-minimum-jump candidates are compared only by field error.
    box(ax, 7, 68, 38, 12, 'Route A   J = 1\nE = eA', TEAL,
        size=14.5)
    box(ax, 55, 68, 38, 12, 'Route B   J = 1\nE = eB', TEAL,
        size=14.5)
    line(ax, [(7, 64), (7, 61), (93, 61), (93, 64)], TEAL, 1.8)
    text(ax, 50, 60, 'First: retain the minimum J', size=14.5,
         color=TEAL)

    box(ax, 7, 42, 38, 12, 'Route C   J = 2\nE = eC', MUTED,
        size=14.5)
    text(ax, 72, 48, 'Higher J:\nnot selected', size=14.5, color=MUTED)
    arrow(ax, (50, 57), (50, 41), TEAL)
    box(ax, 12, 31, 76, 10, 'Then: choose min(eA, eB)', GREEN,
        size=15)

    arrow(ax, (50, 31), (50, 26), GREEN)
    box(ax, 4, 16, 92, 10,
        'After smoothing: recheck turns,\nclearance, jumps, current and field',
        INK, size=14.5)


DRAWERS = {11: draw11, 12: draw12}


if __name__ == '__main__':
    for number, draw in DRAWERS.items():
        save_single(number, draw)
