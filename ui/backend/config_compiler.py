from __future__ import annotations

import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .declarations import DeclarationError, DeclarationStore


SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class ConfigCompileError(DeclarationError):
    """Raised when a frontend request cannot be compiled."""


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ConfigCompileError(f"Cannot assign nested configuration path {path!r}")
        cursor = next_value
    cursor[parts[-1]] = value


def _validate_instance_name(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ConfigCompileError(f"{location} must be a string")
    value = value.strip()
    if not value or len(value) > 80:
        raise ConfigCompileError(f"{location} must contain 1 to 80 characters")
    if any(character in value for character in ("/", "\\", "\0")) or value in {".", ".."}:
        raise ConfigCompileError(f"{location} contains unsafe path characters")
    return value


class ConfigCompiler:
    def __init__(
        self,
        declarations: DeclarationStore,
        project_root: str | Path,
        runtime_dir: str | Path,
    ):
        self.declarations = declarations
        self.project_root = Path(project_root).resolve()
        self.runtime_dir = Path(runtime_dir).resolve()

    def compile(self, request: Any, run_id: str) -> tuple[DictConfig, dict[str, Any]]:
        if not isinstance(request, dict):
            raise ConfigCompileError("Request body must be an object")
        models = request.get("models")
        datasets = request.get("datasets")
        if not isinstance(models, list) or not models:
            raise ConfigCompileError("At least one model is required")
        if not isinstance(datasets, list) or not datasets:
            raise ConfigCompileError("At least one dataset is required")

        config = copy.deepcopy(self.declarations.fixed_config)
        config["models"] = self._compile_entities(models, "model")
        config["datasets"] = self._compile_entities(datasets, "dataset")

        execution = self.declarations.resolve_execution(request.get("execution", {}))
        save_result = execution.pop("save_result")
        for spec in self.declarations.execution_parameters:
            key = spec["key"]
            if spec.get("scope", "config") == "manager":
                continue
            path = spec.get("path")
            if not path:
                raise ConfigCompileError(f"Execution parameter {key!r} has no config path")
            _set_path(config, path, execution[key])

        requested_name = str(config.get("run_name", "")).strip()
        if requested_name and not SAFE_RUN_NAME.fullmatch(requested_name):
            raise ConfigCompileError(
                "run_name must start with an ASCII letter or digit and contain only letters, digits, '.', '_' or '-'"
            )
        generated_name = f"ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id[:6]}"
        config["run_name"] = requested_name or generated_name

        if save_result:
            save_dir = Path(str(config["save_dir"]))
            if save_dir.is_absolute():
                raise ConfigCompileError("Persistent save_dir must be project-relative")
            resolved_save_dir = (self.project_root / save_dir).resolve()
            if self.project_root not in resolved_save_dir.parents and resolved_save_dir != self.project_root:
                raise ConfigCompileError("Persistent save_dir escapes the project directory")
        else:
            resolved_save_dir = self.runtime_dir / "ephemeral_results"
            config["save_dir"] = str(resolved_save_dir)
            config["run_name"] = run_id
            config["override"] = True

        result_path = resolved_save_dir / str(config["run_name"])
        metadata = {
            "save_result": bool(save_result),
            "run_name": str(config["run_name"]),
            "result_path": str(result_path),
            "model_names": [item["model_name"] for item in config["models"]],
            "dataset_names": [item["name"] for item in config["datasets"]],
        }
        return OmegaConf.create(config), metadata

    def _compile_entities(self, entries: list[Any], kind: str) -> list[dict[str, Any]]:
        compiled = []
        names = set()
        for index, entry in enumerate(entries):
            location = f"{kind}s[{index}]"
            if not isinstance(entry, dict):
                raise ConfigCompileError(f"{location} must be an object")
            declaration_id = entry.get("declaration_id")
            declaration = (
                self.declarations.model(declaration_id)
                if kind == "model"
                else self.declarations.dataset(declaration_id)
            )
            instance_name = _validate_instance_name(
                entry.get("instance_name", declaration["default_instance_name"]),
                f"{location}.instance_name",
            )
            if instance_name in names:
                raise ConfigCompileError(f"Duplicate {kind} instance name {instance_name!r}")
            names.add(instance_name)
            parameters = self.declarations.resolve_parameters(
                declaration["parameters"], entry.get("parameters", {}), f"{location}.parameters"
            )
            target = {"_target_": declaration["target"], **parameters}
            key = "model_name" if kind == "model" else "name"
            compiled.append({key: instance_name, "target": target})
        return compiled

