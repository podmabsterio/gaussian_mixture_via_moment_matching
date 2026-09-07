from __future__ import annotations

import copy
import importlib
import json
import math
from pathlib import Path
from typing import Any


SUPPORTED_TYPES = {
    "integer",
    "number",
    "boolean",
    "string",
    "enum",
    "integer_array",
    "number_array",
}


class DeclarationError(ValueError):
    """Raised when a declaration or submitted value is invalid."""


def import_target(path: str) -> Any:
    module_name, separator, attribute = path.rpartition(".")
    if not separator:
        raise DeclarationError(f"Invalid target {path!r}")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise DeclarationError(f"Cannot import target {path!r}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeclarationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DeclarationError(f"{path.name} must contain a JSON object")
    if value.get("schema_version") != 1:
        raise DeclarationError(f"{path.name}: unsupported schema_version")
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_value(value: Any, spec: dict[str, Any], location: str) -> Any:
    if value is None:
        if spec.get("nullable", False):
            return None
        raise DeclarationError(f"{location} cannot be null")

    kind = spec["type"]
    if kind == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise DeclarationError(f"{location} must be an integer")
    elif kind == "number":
        if not _is_number(value):
            raise DeclarationError(f"{location} must be a finite number")
        value = float(value)
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise DeclarationError(f"{location} must be a boolean")
    elif kind == "string":
        if not isinstance(value, str):
            raise DeclarationError(f"{location} must be a string")
        if len(value) > spec.get("maximum_length", 10_000):
            raise DeclarationError(f"{location} is too long")
    elif kind == "enum":
        allowed = [option["value"] for option in spec["options"]]
        if not any(value == option and type(value) is type(option) for option in allowed):
            raise DeclarationError(f"{location} must be one of {allowed!r}")
    elif kind in {"integer_array", "number_array"}:
        if not isinstance(value, list):
            raise DeclarationError(f"{location} must be an array")
        if len(value) < spec.get("min_items", 0):
            raise DeclarationError(f"{location} contains too few values")
        if len(value) > spec.get("max_items", 100_000):
            raise DeclarationError(f"{location} contains too many values")
        item_spec = copy.deepcopy(spec)
        item_spec["type"] = "integer" if kind == "integer_array" else "number"
        item_spec["nullable"] = False
        value = [validate_value(item, item_spec, f"{location}[{index}]") for index, item in enumerate(value)]
    else:  # guarded while loading declarations
        raise DeclarationError(f"{location} has unsupported type {kind!r}")

    values = value if isinstance(value, list) else [value]
    if kind in {"integer", "number", "integer_array", "number_array"}:
        for item in values:
            if "minimum" in spec and item < spec["minimum"]:
                raise DeclarationError(f"{location} must be at least {spec['minimum']}")
            if "maximum" in spec and item > spec["maximum"]:
                raise DeclarationError(f"{location} must be at most {spec['maximum']}")
            if "exclusive_minimum" in spec and item <= spec["exclusive_minimum"]:
                raise DeclarationError(f"{location} must be greater than {spec['exclusive_minimum']}")
            if "exclusive_maximum" in spec and item >= spec["exclusive_maximum"]:
                raise DeclarationError(f"{location} must be less than {spec['exclusive_maximum']}")
    return value


def _validate_parameter(spec: Any, location: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise DeclarationError(f"{location} must be an object")
    for key in ("key", "display_name", "description", "type", "default"):
        if key not in spec:
            raise DeclarationError(f"{location}.{key} is required")
    if not isinstance(spec["key"], str) or not spec["key"]:
        raise DeclarationError(f"{location}.key must be a non-empty string")
    if spec["type"] not in SUPPORTED_TYPES:
        raise DeclarationError(f"{location}.type is unsupported")
    for flag in ("advanced", "hidden", "nullable"):
        if flag in spec and not isinstance(spec[flag], bool):
            raise DeclarationError(f"{location}.{flag} must be boolean")
    if spec["type"] == "enum":
        options = spec.get("options")
        if not isinstance(options, list) or not options:
            raise DeclarationError(f"{location}.options must be a non-empty array")
        for index, option in enumerate(options):
            if not isinstance(option, dict) or "value" not in option or "display_name" not in option:
                raise DeclarationError(f"{location}.options[{index}] is invalid")
    validate_value(spec["default"], spec, f"{location}.default")
    return spec


def _validate_metric(spec: Any, location: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise DeclarationError(f"{location} must be an object")
    for key in ("id", "target", "display_name", "description"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise DeclarationError(f"{location}.{key} must be a non-empty string")
    if spec.get("direction", "minimize") not in {"minimize", "maximize", "neutral"}:
        raise DeclarationError(
            f"{location}.direction must be 'minimize', 'maximize', or 'neutral'"
        )
    if "track_during_fit" in spec and not isinstance(spec["track_during_fit"], bool):
        raise DeclarationError(f"{location}.track_during_fit must be boolean")
    if "format" in spec and not isinstance(spec["format"], str):
        raise DeclarationError(f"{location}.format must be a string")
    arguments = spec.get("arguments", {})
    if not isinstance(arguments, dict):
        raise DeclarationError(f"{location}.arguments must be an object")
    import_target(spec["target"])
    return spec


class DeclarationStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        models_doc = _read_json(self.directory / "models_declaration.json")
        datasets_doc = _read_json(self.directory / "datasets_declaration.json")
        execution_doc = _read_json(self.directory / "execution_parameters.json")

        self.models = self._load_entities(models_doc.get("models"), "models")
        parameter_sets = datasets_doc.get("parameter_sets", {})
        if not isinstance(parameter_sets, dict):
            raise DeclarationError("datasets.parameter_sets must be an object")
        validated_sets: dict[str, list[dict[str, Any]]] = {}
        for name, parameters in parameter_sets.items():
            validated_sets[name] = self._load_parameters(parameters, f"parameter_sets.{name}")
        self.datasets = self._load_entities(
            datasets_doc.get("datasets"), "datasets", validated_sets
        )
        self.execution_parameters = self._load_parameters(
            execution_doc.get("parameters"), "execution.parameters"
        )
        self.quick_parameters = self._load_parameters(
            execution_doc.get("quick_parameters", []), "execution.quick_parameters"
        )
        metrics = execution_doc.get("metrics")
        if not isinstance(metrics, list) or not metrics:
            raise DeclarationError("execution.metrics must be a non-empty array")
        self.metrics = [
            _validate_metric(spec, f"execution.metrics[{index}]")
            for index, spec in enumerate(metrics)
        ]
        metric_ids = [spec["id"] for spec in self.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise DeclarationError("execution.metrics contains duplicate ids")

        self.fixed_config = copy.deepcopy(execution_doc.get("fixed_config"))
        if not isinstance(self.fixed_config, dict):
            raise DeclarationError("execution.fixed_config must be an object")
        self.fixed_config["metrics"] = [
            {"_target_": spec["target"], **copy.deepcopy(spec.get("arguments", {}))}
            for spec in self.metrics
        ]

        self._model_index = {item["id"]: item for item in self.models}
        self._dataset_index = {item["id"]: item for item in self.datasets}
        self._execution_index = {item["key"]: item for item in self.execution_parameters}

    def _load_parameters(self, value: Any, location: str) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise DeclarationError(f"{location} must be an array")
        parameters = [_validate_parameter(spec, f"{location}[{index}]") for index, spec in enumerate(value)]
        keys = [spec["key"] for spec in parameters]
        if len(keys) != len(set(keys)):
            raise DeclarationError(f"{location} contains duplicate parameter keys")
        return parameters

    def _load_entities(
        self,
        value: Any,
        location: str,
        parameter_sets: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or not value:
            raise DeclarationError(f"{location} must be a non-empty array")
        entities = []
        ids = set()
        for index, original in enumerate(value):
            item_location = f"{location}[{index}]"
            if not isinstance(original, dict):
                raise DeclarationError(f"{item_location} must be an object")
            item = copy.deepcopy(original)
            for key in ("id", "display_name", "description", "target", "default_instance_name"):
                if not isinstance(item.get(key), str) or not item[key]:
                    raise DeclarationError(f"{item_location}.{key} must be a non-empty string")
            if item["id"] in ids:
                raise DeclarationError(f"Duplicate {location} id {item['id']!r}")
            ids.add(item["id"])
            parameters: list[dict[str, Any]] = []
            for set_name in item.get("parameter_sets", []):
                if not parameter_sets or set_name not in parameter_sets:
                    raise DeclarationError(f"{item_location} references unknown parameter set {set_name!r}")
                parameters.extend(copy.deepcopy(parameter_sets[set_name]))
            parameters.extend(self._load_parameters(item.get("parameters", []), f"{item_location}.parameters"))
            keys = [spec["key"] for spec in parameters]
            if len(keys) != len(set(keys)):
                raise DeclarationError(f"{item_location} has duplicate parameter keys")
            item["parameters"] = parameters
            item.pop("parameter_sets", None)
            import_target(item["target"])
            entities.append(item)
        return entities

    def public_bundle(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "models": copy.deepcopy(self.models),
            "datasets": copy.deepcopy(self.datasets),
            "execution": {
                "parameters": copy.deepcopy(self.execution_parameters),
                "quick_parameters": copy.deepcopy(self.quick_parameters),
            },
            "metrics": copy.deepcopy(self.metrics),
        }

    def model(self, entity_id: str) -> dict[str, Any]:
        try:
            return self._model_index[entity_id]
        except KeyError as exc:
            raise DeclarationError(f"Unknown model declaration {entity_id!r}") from exc

    def dataset(self, entity_id: str) -> dict[str, Any]:
        try:
            return self._dataset_index[entity_id]
        except KeyError as exc:
            raise DeclarationError(f"Unknown dataset declaration {entity_id!r}") from exc

    def resolve_parameters(
        self,
        specs: list[dict[str, Any]],
        submitted: Any,
        location: str,
    ) -> dict[str, Any]:
        if submitted is None:
            submitted = {}
        if not isinstance(submitted, dict):
            raise DeclarationError(f"{location} must be an object")
        index = {spec["key"]: spec for spec in specs}
        unknown = set(submitted) - set(index)
        if unknown:
            raise DeclarationError(f"{location} contains unknown parameters: {sorted(unknown)!r}")
        hidden_submitted = [key for key in submitted if index[key].get("hidden", False)]
        if hidden_submitted:
            raise DeclarationError(f"{location} cannot override hidden parameters: {hidden_submitted!r}")

        resolved = {}
        for spec in specs:
            key = spec["key"]
            raw = spec["default"] if spec.get("hidden", False) or key not in submitted else submitted[key]
            resolved[key] = validate_value(raw, spec, f"{location}.{key}")

        for spec in specs:
            value = resolved[spec["key"]]
            if value is None:
                continue
            for other in spec.get("conflicts_with", []):
                if resolved.get(other) is not None:
                    raise DeclarationError(
                        f"{location}.{spec['key']} conflicts with {location}.{other}"
                    )
        return resolved

    def resolve_execution(self, submitted: Any) -> dict[str, Any]:
        return self.resolve_parameters(self.execution_parameters, submitted, "execution")

    def resolve_quick_execution(self, submitted: Any) -> dict[str, Any]:
        return self.resolve_parameters(
            self.quick_parameters,
            submitted,
            "quick_execution",
        )
