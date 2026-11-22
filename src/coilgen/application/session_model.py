"""Shared project/channel transitions and immutable undo records for both clients."""

from dataclasses import dataclass

from .assembly import ProjectAssembly


@dataclass(frozen=True)
class ModelSnapshot:
    assembly: ProjectAssembly | None = None
    channel_id: str | None = None

    @property
    def project(self):
        return self.assembly.channel(self.channel_id).project if self.assembly is not None and self.channel_id else None

    def with_project(self, project, channel_id=None):
        selected = channel_id or self.channel_id
        if self.assembly is None:
            assembly = ProjectAssembly.new(project.name, (project,))
            return ModelSnapshot(assembly, assembly.channels[0].id)
        channel = self.assembly.channel(selected)
        assembly = (
            self.assembly if channel.project is project else self.assembly.update_channel(selected, project=project)
        )
        return ModelSnapshot(assembly, selected)


@dataclass(frozen=True)
class ModelEdit:
    before: ModelSnapshot
    after: ModelSnapshot


class SessionModel:
    """Own the active assembly once; adapters retain their native notification mechanisms."""

    def __init__(self):
        self.snapshot = ModelSnapshot()

    @property
    def project(self):
        return self.snapshot.project

    @property
    def assembly(self):
        return self.snapshot.assembly

    @property
    def channel_id(self):
        return self.snapshot.channel_id

    def install(self, assembly, channel_id=None):
        ids = [channel.id for channel in assembly.channels] if assembly is not None else []
        selected = (
            channel_id if channel_id in ids else self.channel_id if self.channel_id in ids else ids[0] if ids else None
        )
        before = self.snapshot
        self.snapshot = ModelSnapshot(assembly, selected)
        return ModelEdit(before, self.snapshot)

    def select(self, channel_id):
        self.assembly.channel(channel_id)
        self.snapshot = ModelSnapshot(self.assembly, channel_id)

    def replace_project(self, project, channel_id=None):
        before = self.snapshot
        self.snapshot = before.with_project(project, channel_id)
        return ModelEdit(before, self.snapshot)

    def duplicate(self, name=None, channel_id=None):
        source = self.assembly.channel(channel_id or self.channel_id)
        if name is None:
            existing = {channel.name.strip().casefold() for channel in self.assembly.channels}
            index = 1
            while True:
                suffix = " copy" if index == 1 else f" copy {index}"
                name = source.name[: 120 - len(suffix)].rstrip() + suffix
                if name.casefold() not in existing:
                    break
                index += 1
        project = source.project.duplicate(name)
        assembly = self.assembly.add_channel(project, name=name)
        return ModelEdit(self.snapshot, ModelSnapshot(assembly, assembly.channels[-1].id))

    def removal(self, channel_id=None, *, select_first=False):
        if len(self.assembly.channels) <= 1:
            raise ValueError("A project keeps at least one channel.")
        assembly = self.assembly.remove_channel(channel_id or self.channel_id)
        ids = [channel.id for channel in assembly.channels]
        selected = self.channel_id if not select_first and self.channel_id in ids else ids[0]
        return ModelEdit(self.snapshot, ModelSnapshot(assembly, selected))


class ModelHistory:
    """Headless undo uses the same before/after records as the Qt undo adapter."""

    def __init__(self):
        self.undo: list[ModelEdit] = []
        self.redo: list[ModelEdit] = []

    def record(self, edit):
        if edit.before.assembly is not None:
            self.undo.append(edit)
            self.redo.clear()

    def step(self, *, redo=False):
        source, destination = (self.redo, self.undo) if redo else (self.undo, self.redo)
        if not source:
            return None
        edit = source.pop()
        destination.append(edit)
        return edit.after if redo else edit.before
