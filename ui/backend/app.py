from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from .config_compiler import ConfigCompileError, ConfigCompiler
from .data_preview import DataPreviewService
from .declarations import DeclarationError, DeclarationStore
from .results_catalog import ResultsCatalog, SavedRunNotFound
from .run_manager import RunManager, RunNotFound
from .quick_experiment import (
    QuickExperimentManager,
    QuickRunConflict,
    QuickRunNotFound,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = Path(__file__).resolve().parents[1]


def create_app(
    declarations: DeclarationStore | None = None,
    manager: RunManager | None = None,
    results_catalog: ResultsCatalog | None = None,
    quick_manager: QuickExperimentManager | None = None,
) -> Flask:
    declarations = declarations or DeclarationStore(UI_ROOT / "declarations")
    manager = manager or RunManager(PROJECT_ROOT, PROJECT_ROOT / ".ui_runtime")
    compiler = ConfigCompiler(declarations, PROJECT_ROOT, manager.runtime_dir)
    preview_service = DataPreviewService(declarations)
    results_catalog = results_catalog or ResultsCatalog(manager.project_root / "results")
    quick_manager = quick_manager or QuickExperimentManager(
        declarations,
        manager.project_root,
        manager.runtime_dir / "quick",
        results_catalog.results_root,
    )

    app = Flask(__name__, static_folder=str(UI_ROOT / "frontend"), static_url_path="")
    app.config["JSON_SORT_KEYS"] = False
    app.extensions["ui_declarations"] = declarations
    app.extensions["ui_run_manager"] = manager
    app.extensions["ui_config_compiler"] = compiler
    app.extensions["ui_data_preview"] = preview_service
    app.extensions["ui_results_catalog"] = results_catalog
    app.extensions["ui_quick_manager"] = quick_manager

    @app.get("/")
    def index() -> Any:
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/quick")
    def quick_page() -> Any:
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.get("/api/declarations")
    def get_declarations() -> Any:
        return jsonify(declarations.public_bundle())

    @app.post("/api/runs")
    def start_run() -> Any:
        body = request.get_json(silent=False)
        run_id = manager.new_id()
        config, metadata = compiler.compile(body, run_id)
        result_path = Path(metadata["result_path"])
        if result_path.exists() and not bool(config.override):
            return jsonify({"error": f"Run directory already exists: {result_path}"}), 409
        return jsonify(manager.start(run_id, config, metadata, body)), 202

    @app.post("/api/data-preview")
    def data_preview() -> Any:
        return jsonify(preview_service.build(request.get_json(silent=False)))

    @app.post("/api/quick-runs")
    def start_quick_run() -> Any:
        return jsonify(quick_manager.start(request.get_json(silent=False))), 202

    @app.get("/api/quick-runs/<run_id>")
    def get_quick_run(run_id: str) -> Any:
        return jsonify(quick_manager.get(run_id))

    @app.post("/api/quick-runs/<run_id>/cancel")
    def cancel_quick_run(run_id: str) -> Any:
        return jsonify(quick_manager.cancel(run_id))

    @app.post("/api/quick-runs/<run_id>/save")
    def save_quick_run(run_id: str) -> Any:
        body = request.get_json(silent=True) or {}
        return jsonify(quick_manager.save(run_id, body.get("run_name")))

    @app.get("/api/quick-runs/<run_id>/events")
    def quick_run_events(run_id: str) -> Any:
        quick_manager.get(run_id)  # fail with 404 before opening the stream
        raw_after = request.args.get("after") or request.headers.get("Last-Event-ID") or "0"
        try:
            after = max(0, int(raw_after))
        except ValueError:
            after = 0

        @stream_with_context
        def stream():
            cursor = after
            while True:
                events, terminal = quick_manager.events_since(run_id, cursor)
                if not events:
                    yield ": keep-alive\n\n"
                    if terminal:
                        return
                    continue
                for event in events:
                    cursor = event["sequence"]
                    yield (
                        f"id: {cursor}\n"
                        f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    )
                if terminal:
                    return

        return Response(
            stream(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs")
    def list_runs() -> Any:
        return jsonify(manager.list())

    @app.get("/api/comparison")
    def comparison() -> Any:
        return jsonify(
            results_catalog.compare(
                run=request.args.get("run") or None,
                dataset=request.args.get("dataset") or None,
                model=request.args.get("model") or None,
                group_by=request.args.get("group_by", "model"),
                statistic=request.args.get("statistic", "mean_std"),
            )
        )

    @app.get("/api/saved-runs")
    def saved_runs() -> Any:
        return jsonify(results_catalog.saved_runs())

    @app.get("/api/saved-runs/<path:run_name>")
    def saved_run(run_name: str) -> Any:
        return jsonify(results_catalog.saved_run(run_name))

    @app.get("/api/runs/<run_id>")
    def get_run(run_id: str) -> Any:
        return jsonify(manager.get(run_id))

    @app.get("/api/runs/<run_id>/results")
    def get_results(run_id: str) -> Any:
        return jsonify(manager.results(run_id))

    @app.post("/api/runs/<run_id>/cancel")
    def cancel_run(run_id: str) -> Any:
        return jsonify(manager.cancel(run_id))

    @app.errorhandler(DeclarationError)
    @app.errorhandler(ConfigCompileError)
    def declaration_error(error: Exception) -> Any:
        return jsonify({"error": str(error)}), 400

    @app.errorhandler(RunNotFound)
    def run_not_found(error: RunNotFound) -> Any:
        return jsonify({"error": f"Run {error.args[0]!r} was not found"}), 404

    @app.errorhandler(SavedRunNotFound)
    def saved_run_not_found(error: SavedRunNotFound) -> Any:
        return jsonify({"error": f"Saved run {error.args[0]!r} was not found"}), 404

    @app.errorhandler(QuickRunNotFound)
    def quick_run_not_found(error: QuickRunNotFound) -> Any:
        return jsonify({"error": f"Quick run {error.args[0]!r} was not found"}), 404

    @app.errorhandler(QuickRunConflict)
    def quick_run_conflict(error: QuickRunConflict) -> Any:
        return jsonify({"error": str(error)}), 409

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)
