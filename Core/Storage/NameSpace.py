import os
from typing import Optional


class Workspace:
    @staticmethod
    def new(working_dir: str, exp_name: str = None) -> "Workspace":
        return Workspace(working_dir, exp_name)

    @staticmethod
    def get_path(working_dir: str, exp_name: Optional[str] = None) -> Optional[str]:
        if exp_name is None:
            return working_dir
        return os.path.join(working_dir, exp_name)

    def __init__(self, working_dir: str, exp_name: str = None):
        self.working_dir: str = working_dir
        self.exp_name: str = exp_name
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

    def make_for(self, namespace: str) -> "Namespace":
        return Namespace(self, namespace)

    def get_load_path(self) -> Optional[str]:
        load_path = self.get_path(self.working_dir, self.exp_name)
        if load_path == self.working_dir and len([x for x in os.scandir(load_path) if x.is_file()]) == 0:
            return None
        return load_path

    def get_save_path(self) -> str:

        save_path = self.get_path(self.working_dir, self.exp_name)

        assert save_path is not None, "Save path cannot be None."

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path)


class Namespace:
    def __init__(self, workspace: Workspace, namespace: Optional[str] = None):
        self.namespace = namespace
        self.workspace = workspace

    def get_load_path(self, resource_name: Optional[str] = None) -> Optional[str]:
        assert self.namespace is not None, "Namespace must be set to get resource load path."
        load_path = self.workspace.get_load_path()
        if load_path is None:
            return None
        if resource_name:
            return os.path.join(load_path, f"{self.namespace}_{resource_name}")
        return os.path.join(load_path, self.namespace)

    def get_save_path(self, resource_name: Optional[str] = None) -> str:
        assert self.namespace is not None, "Namespace must be set to get resource save path."
        if resource_name:
            return os.path.join(self.workspace.get_save_path(), f"{self.namespace}_{resource_name}")
        return os.path.join(self.workspace.get_save_path(), self.namespace)
