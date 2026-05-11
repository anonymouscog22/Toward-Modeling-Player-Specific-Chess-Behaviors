"""Utilities to generate LaTeX tables summarizing datasets, models and results.

This module implements functions that produce LaTeX-ready tables for the
paper: dataset overview, AutoEncoder architecture, training hyperparameters,
accuracy comparisons and Jensen-Shannon Divergence stability tables.
"""

from pathlib import Path

import numpy as np
import polars as pl

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def generate_latex_table(config: Config) -> None:
    """Generate a LaTeX table summarizing the selected players' metadata.

    The produced table contains player name, the average year of games and the
    number of games contributed by each player. The resulting LaTeX fragment is
    written to `config.paths.table_latex_path`.
    """

    df = pl.read_parquet(config.paths.player_stats_path)

    df_sorted = df.select(
        [pl.col("player_name"), pl.col("mean_year"), pl.col("n_games")]
    ).sort("player_name")

    latex_rows = []
    for row in df_sorted.iter_rows():
        name, year, games = row
        # Construct LaTeX table row safely and avoid escaping errors
        row_str = " & ".join([str(name), str(year), str(games)]) + " \\\\"
        latex_rows.append(row_str)

    table_body = "\n".join(latex_rows)

    latex_template = (
        "\\begin{table}[!t]\n"
        "\\renewcommand{\\arraystretch}{1.3}\n"
        "\\caption{Overview of selected chess champions}\n"
        "\\label{tab:dataset}\n"
        "\\centering\n"
        "\\begin{tabular}{l c c}\n"
        "\\hline\n"
        "\\bfseries Player & \\bfseries Average game's year & \\bfseries Games \\\\n"
        "\\hline\\hline\n"
        f"{table_body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    with open(config.paths.table_latex_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info("LaTeX table generated and saved to %s", config.paths.table_latex_path)


def generate_ae_latex_table(config: Config) -> None:
    """Generate a LaTeX table describing the AutoEncoder architecture and training parameters."""

    # Extract relevant hyperparameters
    lr = config.autoencoder.learning_rate
    batch_size = config.autoencoder.batch_size
    epochs = config.autoencoder.epochs
    latent_dim = config.autoencoder.latent_dim

    out_path = config.paths.ae_table_latex_path

    # Use a compact notation for layer shapes in the table
    latex_template = f"""\\begin{{table}}[!t]
                         \\renewcommand{{\\arraystretch}}{{1.3}}
                         \\caption{{Architecture and training parameters of the AutoEncoder}}
                         \\label{{tab:ae}}
                         \\centering
                         \\begin{{tabular}}{{@{{}}l l@{{}}}}
                         \\hline
                         \\bfseries Parameter & \\bfseries Value \\\\n                         \\hline\\hline
                         Learning Rate & {lr} \\\\n                         Batch Size & {batch_size} \\\\n                         Epochs & {epochs} \\\\n                         Latent Dimension & {latent_dim} \\\\n                         \\hline
                         Encoder layers & $2304 \\rightarrow 1024 \\rightarrow 512 \\rightarrow 256 \\rightarrow {latent_dim}$ \\\\
                         Decoder layers & ${latent_dim} \\rightarrow 256 \\rightarrow 512 \\rightarrow 1024 \\rightarrow 2304$ \\\\
                         Hidden Activations & ReLU \\\\n                         Output Activation & Sigmoid \\\\n                         \\hline
                         \\end{{tabular}}
                         \\end{{table}}"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info(f"AutoEncoder LaTeX table generated and saved to {out_path}")


def generate_accuracy_latex_table(config: Config) -> None:
    """Generate a LaTeX table reporting per-player move prediction accuracies.

    The table contains absolute accuracies for each method and the delta
    relative to the Maia-2 baseline presented in parentheses.
    """

    # Load per-player accuracies
    df_acc = pl.read_parquet(config.paths.accuracy_path).sort("player_name")

    # Compute global means (average across players)
    overall_baseline = df_acc.select(pl.col("baseline_accuracy").mean()).item()
    overall_custom = df_acc.select(pl.col("custom_accuracy").mean()).item()
    overall_mcts = df_acc.select(pl.col("mcts_accuracy").mean()).item()

    # Formatage des lignes du tableau
    latex_rows = []
    for row in df_acc.iter_rows(named=True):
        name = row["player_name"]

        # Baseline en valeur absolue
        acc_b = f"{row['baseline_accuracy'] * 100:.1f}\\%"

        # Custom : Absolu + Delta
        abs_c = row["custom_accuracy"] * 100
        delta_c = (row["custom_accuracy"] - row["baseline_accuracy"]) * 100
        acc_c = f"{abs_c:.1f}\\% ({delta_c:+.1f}\\%)"

        # MCTS : Absolu + Delta
        abs_m = row["mcts_accuracy"] * 100
        delta_m = (row["mcts_accuracy"] - row["baseline_accuracy"]) * 100
        acc_m = f"{abs_m:.1f}\\% ({delta_m:+.1f}\\%)"

        latex_rows.append(
            f"                         {name} & {acc_b} & {acc_c} & {acc_m} \\\\"
        )

    # Append a row with global averages
    latex_rows.append("                         \\hline")

    # Compute deltas for the global averages
    avg_abs_c = overall_custom * 100
    avg_delta_c = (overall_custom - overall_baseline) * 100
    avg_c_str = f"{avg_abs_c:.1f}\\% ({avg_delta_c:+.1f}\\%)"

    avg_abs_m = overall_mcts * 100
    avg_delta_m = (overall_mcts - overall_baseline) * 100
    avg_m_str = f"{avg_abs_m:.1f}\\% ({avg_delta_m:+.1f}\\%)"

    latex_rows.append(
        f"                         \\bfseries Average & \\bfseries {overall_baseline * 100:.1f}\\% & \\bfseries {avg_c_str} & \\bfseries {avg_m_str} \\\\"
    )

    table_body = "\n".join(latex_rows)

    # The header omits a separate delta symbol since differences are presented as (+-%) next to the absolute values
    latex_template = f"""\\begin{{table}}[!t]
                         \\renewcommand{{\\arraystretch}}{{1.3}}
                         \\caption{{Move-accuracy of the different models on the test set. Values in parentheses indicate the difference relative to Maia-2.}}
                         \\label{{tab:move_accuracy}}
                         \\centering
                         \\begin{{tabular}}{{l c c c}}
                         \\hline
                         \\bfseries Player & \\bfseries Maia-2 & \\bfseries Maia-2 FT & \\bfseries Maia-2 FT + MCTS \\\\n                         \\hline\\hline
{table_body}
                         \\hline
                         \\end{{tabular}}
                         \\end{{table}}"""

    # Persist the accuracy table to the configured output location
    out_path = config.paths.accuracy_table_latex_path

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info("Accuracy LaTeX table generated and saved to %s", out_path)


def generate_training_hyperparameters_latex_table(config: Config) -> None:
    """Generate a LaTeX table summarizing training hyperparameters for player embedding training."""

    # Extract hyperparameters from config.player_training with sensible defaults
    optimizer = getattr(config.player_training, "optimizer", "Adam")
    lr = getattr(config.player_training, "learning_rate", 1e-4)
    batch_size = getattr(config.player_training, "batch_size", 512)
    epochs = getattr(config.player_training, "epochs", 10)
    loss = getattr(config.player_training, "loss_function", "Cross-Entropy")

    # Format learning rate in scientific LaTeX notation when small (e.g. 1 \times 10^{-4})
    if isinstance(lr, (float, int)) and lr < 0.01:
        base, exp = f"{lr:.0e}".split("e")
        lr_str = f"${base} \\times 10^{{{int(exp)}}}$"
    else:
        lr_str = str(lr)

    # Output path (fallback if not configured explicitly)
    out_path = getattr(
        config.paths,
        "hyperparameters_table_latex_path",
        str(Path(config.paths.table_latex_path).parent / "hyperparameters_table.tex"),
    )

    latex_template = f"""\\begin{{table}}[!t]
                         \\renewcommand{{\\arraystretch}}{{1.3}}
                         \\caption{{Training Hyperparameters}}
                         \\label{{tab:hyperparameters}}
                         \\centering
                         \\begin{{tabular}}{{l c}}
                         \\hline
                         \\bfseries Hyperparameter & \\bfseries Value \\\\n                         \\hline\\hline
                         Optimizer & {optimizer} \\\\n                         Learning Rate & {lr_str} \\\\n                         Batch Size & {batch_size} \\\\n                         Epochs & {epochs} \\\\n                         Loss Function & {loss} \\\\n                         \\hline
                         \\end{{tabular}}
                         \\end{{table}}"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info(
        "Training hyperparameters LaTeX table generated and saved to %s", out_path
    )


def generate_jsd_stability_table(config: Config, show_ci: bool = False) -> None:
    """Generate a LaTeX table with per-player diagonal JSD and either std devs or CI95.

    Args:
        config: project configuration containing paths and metadata.
        show_ci: if True, display 95% confidence intervals (ci_lower_95/ci_upper_95)
                 when available; otherwise display std dev (bs_std or similar).

    Matches the style of `generate_accuracy_latex_table`:
    - Per-player rows with TEST (reference) and three Maia columns
    - An "Average" row at the bottom with bolded mean values and mean std devs/CI

    For each method the table shows the absolute JSD value and, in
    parentheses, either the standard deviation (default) or the 95% CI
    (if `show_ci=True` and CI columns are present).
    """

    methods = ["maia2", "maia2_ft", "maia2_ft_mcts"]
    reference = "umap"

    # Human-readable, title-cased labels to match the accuracy table
    method_labels = ["Maia-2", "Maia-2 FT", "Maia-2 FT + MCTS"]

    def _load_diag(method_name: str):
        """Return four dicts: distances, stds, ci_lower, ci_upper indexed by player.

        Tries to load a parquet with cross distances first (column name
        "distance" expected). If a std-like column exists (e.g. "bs_std",
        "distance_std", "std"), it is extracted; if CI columns exist
        (e.g. "ci_lower_95","ci_upper_95") they are also extracted.
        Otherwise NaN is used for missing items.
        """

        std_candidates = ["bs_std", "distance_std", "std", "std_dev", "sigma"]
        ci_lower_candidates = ["ci_lower_95", "ci_lower", "ci95_lower"]
        ci_upper_candidates = ["ci_upper_95", "ci_upper", "ci95_upper"]

        def _extract_from_rows(rows, player_key: str):
            dists = {}
            stds = {}
            ci_lowers = {}
            ci_uppers = {}
            for r in rows:
                player = r.get(player_key)
                if player is None:
                    continue
                dist = r.get("distance", float("nan"))

                s = float("nan")
                for col in std_candidates:
                    if col in r and r[col] == r[col]:
                        s = r[col]
                        break

                ci_l = float("nan")
                ci_u = float("nan")
                for col in ci_lower_candidates:
                    if col in r and r[col] == r[col]:
                        ci_l = r[col]
                        break
                for col in ci_upper_candidates:
                    if col in r and r[col] == r[col]:
                        ci_u = r[col]
                        break

                dists[player] = dist
                stds[player] = s
                ci_lowers[player] = ci_l
                ci_uppers[player] = ci_u
            return dists, stds, ci_lowers, ci_uppers

        try:
            p = config.paths.get_cross_distances_path(method_name, config.jsd.kde)
            df = pl.read_parquet(p)
            if df is not None and df.height > 0:
                return _extract_from_rows(df.iter_rows(named=True), "player")
        except Exception:
            pass

        try:
            p = config.paths.get_full_cross_matrix_path(method_name, config.jsd.kde)
            df = pl.read_parquet(p)
            if df is not None and df.height > 0:
                diag = df.filter(pl.col("p_train") == pl.col("p_test"))
                return _extract_from_rows(diag.iter_rows(named=True), "p_train")
        except Exception:
            pass

        return {}, {}, {}, {}

    ref_dists, ref_stds, ref_ci_lowers, ref_ci_uppers = _load_diag(reference)
    method_vals = {m: _load_diag(m) for m in methods}

    # Determine player ordering: alphabetical (case-insensitive) for reproducible display
    if ref_dists:
        players_order = sorted(list(ref_dists.keys()), key=lambda s: s.lower())
    else:
        union_players = set()
        for dists, stds, ci_l, ci_u in method_vals.values():
            union_players.update(dists.keys())
        if union_players:
            players_order = sorted(union_players, key=lambda s: s.lower())
        else:
            players_order = sorted(
                list(config.data.players.values()), key=lambda s: s.lower()
            )

    # Build table rows
    latex_rows = []
    for player in players_order:
        ref = ref_dists.get(player, float("nan"))
        ref_s = ref_stds.get(player, float("nan"))
        ref_ci_l = ref_ci_lowers.get(player, float("nan"))
        ref_ci_u = ref_ci_uppers.get(player, float("nan"))

        if ref == ref:
            if show_ci and ref_ci_l == ref_ci_l and ref_ci_u == ref_ci_u:
                ref_str = f"{ref:.3f} ({ref_ci_l:.3f}-{ref_ci_u:.3f})"
            elif ref_s == ref_s:
                ref_str = f"${ref:.3f}\\pm{ref_s:.3f}$"
            else:
                ref_str = f"{ref:.3f} (N/A)"
        else:
            ref_str = "N/A (N/A)"

        cols = [ref_str]
        for m in methods:
            dists, stds, ci_lowers, ci_uppers = method_vals[m]
            v = dists.get(player, float("nan"))
            s = stds.get(player, float("nan"))
            ci_l = ci_lowers.get(player, float("nan"))
            ci_u = ci_uppers.get(player, float("nan"))

            if v == v:
                if show_ci and ci_l == ci_l and ci_u == ci_u:
                    cols.append(f"{v:.3f} ({ci_l:.3f}-{ci_u:.3f})")
                elif s == s:
                    cols.append(f"${v:.3f}\\pm{s:.3f}$")
                else:
                    cols.append(f"{v:.3f} (N/A)")
            else:
                cols.append("N/A (N/A)")

        row = "                         " + player + " & " + " & ".join(cols) + " \\\\"
        latex_rows.append(row)

    # Compute averages across the same player order (ignore NaNs)
    def _mean_for_players(d: dict) -> float:
        vals = [d.get(p) for p in players_order]
        nums = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        return float(np.mean(nums)) if nums else float("nan")

    # Mean of stds across players for each method
    def _mean_std_for_players(sdict: dict) -> float:
        vals = [sdict.get(p) for p in players_order]
        nums = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        return float(np.mean(nums)) if nums else float("nan")

    # Mean of CI endpoints across players
    def _mean_ci_for_players(cdict: dict) -> float:
        vals = [cdict.get(p) for p in players_order]
        nums = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        return float(np.mean(nums)) if nums else float("nan")

    overall_ref = _mean_for_players(ref_dists)
    overall_ref_std = _mean_std_for_players(ref_stds)
    overall_ref_ci_l = _mean_ci_for_players(ref_ci_lowers)
    overall_ref_ci_u = _mean_ci_for_players(ref_ci_uppers)
    overall_methods = {m: _mean_for_players(method_vals[m][0]) for m in methods}
    overall_methods_std = {m: _mean_std_for_players(method_vals[m][1]) for m in methods}
    overall_methods_ci_l = {m: _mean_ci_for_players(method_vals[m][2]) for m in methods}
    overall_methods_ci_u = {m: _mean_ci_for_players(method_vals[m][3]) for m in methods}

    # Average row values (formatted) — use 2 decimals to save horizontal space
    if overall_ref == overall_ref:
        if (
            show_ci
            and overall_ref_ci_l == overall_ref_ci_l
            and overall_ref_ci_u == overall_ref_ci_u
        ):
            avg_ref_str = (
                f"{overall_ref:.3f} ({overall_ref_ci_l:.3f}-{overall_ref_ci_u:.3f})"
            )
        elif overall_ref_std == overall_ref_std:
            avg_ref_str = f"${overall_ref:.3f}\\pm{overall_ref_std:.3f}$"
        else:
            avg_ref_str = f"{overall_ref:.3f} (N/A)"
    else:
        avg_ref_str = "N/A (N/A)"
    avg_method_strs = []
    for m in methods:
        mv = overall_methods.get(m, float("nan"))
        ms = overall_methods_std.get(m, float("nan"))
        mci_l = overall_methods_ci_l.get(m, float("nan"))
        mci_u = overall_methods_ci_u.get(m, float("nan"))
        if mv == mv:
            if show_ci and mci_l == mci_l and mci_u == mci_u:
                avg_method_strs.append(f"{mv:.3f} ({mci_l:.3f}-{mci_u:.3f})")
            elif ms == ms:
                avg_method_strs.append(f"${mv:.3f}\\pm{ms:.3f}$")
            else:
                avg_method_strs.append(f"{mv:.3f} (N/A)")
        else:
            avg_method_strs.append("N/A (N/A)")

    # Append average row in bold, matching the accuracy table style
    latex_rows.append("                         \\hline")
    avg_row = (
        "                         \\bfseries Average & \\bfseries "
        + avg_ref_str
        + " & \\bfseries "
        + avg_method_strs[0]
        + " & \\bfseries "
        + avg_method_strs[1]
        + " & \\bfseries "
        + avg_method_strs[2]
        + " \\\\"
    )
    latex_rows.append(avg_row)

    # Use human-readable labels for the header
    header = " & ".join(["Player", "TEST"] + method_labels)
    table_body = "\n".join(latex_rows)

    # Build LaTeX table using a raw f-string to avoid escape confusion
    if show_ci:
        caption = "Jensen-Shannon divergence compared to the training set for each player and each model. Values in parentheses indicate the 95% confidence interval."
    else:
        caption = "Jensen-Shannon divergence compared to the training set for each player and each model. Values following the $\\pm$ indicate the standard deviation."

    latex_template = rf"""\begin{{table}}[!t]
\renewcommand{{\arraystretch}}{{1.3}}
\caption{{{caption}}}
\label{{tab:jsd_stability}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{l c c c c}}
\hline
\bfseries {header} \\
\hline\hline
{table_body}
\hline
\end{{tabular}}
\end{{table}}"""

    out_path = config.paths.jsd_table_latex_path
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_template)

    logger.info("JSD stability LaTeX table saved to %s", out_path)


def generate_all_tables(config: Config) -> None:
    """Generate all LaTeX tables for the paper."""
    generate_ae_latex_table(config)
    generate_latex_table(config)
    generate_training_hyperparameters_latex_table(config)
    generate_accuracy_latex_table(config)
    generate_jsd_stability_table(config)
