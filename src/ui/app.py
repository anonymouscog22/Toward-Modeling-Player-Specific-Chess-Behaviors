"""Flask web interface for interactive move prediction and exploration.

This module exposes `create_app` which constructs a Flask application embedding
an instantiated `MaiaEngine`. The application provides endpoints for rendering
an interactive board UI and for requesting model-predicted moves via MCTS.
"""

from flask import Flask, jsonify, render_template, request

from src.core.config import Config
from src.core.utils import getLogger
from src.models.maia import MaiaEngine

logger = getLogger()


def create_app(config: Config) -> Flask:
    """Create and configure the Flask application using a Maia engine.

    The Flask instance uses a local `templates` directory for rendering the web
    interface. The Maia engine is instantiated at startup and used to serve
    move-prediction requests via a dedicated endpoint.
    """
    # Templates directory is relative to this module
    app = Flask(__name__, template_folder="templates")

    logger.info("Initializing Maia engine for web interface (loading model)...")
    engine = MaiaEngine(config)

    @app.route("/")
    def index():
        # Dynamically obtain player and Elo options from the configuration
        players = list(config.data.players.values())
        _, elo_dict, _ = engine.prepare
        standard_elos = list(elo_dict.keys())

        return render_template(
            "index.html", players=players, standard_elos=standard_elos
        )

    @app.route("/get-move", methods=["POST"])
    def get_move():
        data = request.get_json()
        try:
            # Use Maia's MCTS-based predictor for move selection
            move_uci, move_dict = engine.predict_mcts(
                fen=data["fen"],
                pgn=data.get("pgn", ""),
                active_elo=data["active_elo"],
                opponent_elo=data["opponent_elo"],
                c_puct=1.5,
                threshold=0.01,
                num_simulations=100,
            )

            return jsonify({"move": move_uci, "move_dict": move_dict})

        except Exception as e:
            logger.error("Error while predicting move for web UI: %s", e)
            return jsonify({"error": str(e)}), 500

    return app


def run_ui(config: Config) -> None:
    """Launch the local Flask web server for interactive exploration.

    The server is launched with `debug=False` by default to avoid double-loading
    heavy PyTorch models during development reloads.
    """
    app = create_app(config)
    logger.info("Starting web interface at http://127.0.0.1:5000")
    # Debug is disabled to prevent double-loading the heavy PyTorch model
    app.run(host="0.0.0.0", port=5000, debug=False)
