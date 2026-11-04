"""Browser workbench: the shared design session served to web browsers.

``python -m coilgen.web`` starts the hub (login, one isolated session per
user); ``python -m coilgen.web.app`` serves a single session directly. Both
drive :class:`coilgen.application.session.WorkbenchSession`, the same core the
desktop workbench uses.
"""
