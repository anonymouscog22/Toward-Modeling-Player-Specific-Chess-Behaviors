"""Module for fetching chess games (PGN) from chessgames.com.

This module provides functions to retrieve HTML pages and download Portable Game Notation
(PGN) files for a set of players defined in the project's configuration. The implementation
employs retry logic with exponential backoff, parallel downloads via a thread pool, and
progress reporting to ensure robust and observable data acquisition.
"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from lxml import html
from tenacity import retry, stop_after_delay, wait_exponential
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


@retry(stop=stop_after_delay(360), wait=wait_exponential(max=60), reraise=True)
def fetch_url(url: str, headers: dict) -> html.HtmlElement:
    """Retrieve and parse the HTML content at the specified URL.

    This function performs an HTTP GET request and returns an lxml HTML tree. It is
    decorated with retry logic (tenacity) to tolerate transient network errors and
    server-side throttling.

    Parameters
    ----------
    url : str
        Target URL to fetch.
    headers : dict
        HTTP headers to include in the request.

    Returns
    -------
    html.HtmlElement
        Parsed HTML document.

    Raises
    ------
    Exception
        If the HTTP response status code is not 200.
    """
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to retrieve URL: {url}")

    return html.fromstring(response.content)


@retry(stop=stop_after_delay(360), wait=wait_exponential(max=60), reraise=True)
def download_pgn(
    pid: str, gid: str, download_url: str, save_dir: Path, headers: dict
) -> None:
    """Download a single PGN file and persist it to disk.

    The function creates a directory for the player (if necessary) and writes the PGN
    content to a file named `{gid}.pgn`. It includes a small randomized delay to help
    mitigate server throttling when many downloads are performed in succession.

    Parameters
    ----------
    pid : str
        Player identifier used to organize saved PGN files.
    gid : str
        Game identifier (used for the filename).
    download_url : str
        Direct URL for downloading the PGN content.
    save_dir : Path
        Base directory where PGNs will be stored.
    headers : dict
        HTTP headers to include in the request.

    Raises
    ------
    Exception
        If the HTTP response status code is not 200.
    """
    player_dir = save_dir / pid
    player_dir.mkdir(parents=True, exist_ok=True)

    file_path = player_dir / f"{gid}.pgn"

    if file_path.exists():
        return

    response = requests.get(download_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to download PGN for game {gid}")

    with open(file_path, "wb") as f:
        f.write(response.content)

    time.sleep(random.uniform(2, 4))


def fetch_chessgames(
    player_id: str, player_name: str, executor: ThreadPoolExecutor, config: Config
) -> None:
    """Iterate through a player's game listing pages and schedule PGN downloads.

    This function walks the paginated game list for a single player and submits download
    tasks to the provided thread pool executor. Progress is reported via a tqdm progress
    bar.

    Parameters
    ----------
    player_id : str
        Identifier of the player whose games will be fetched.
    player_name : str
        Display name used in progress reporting.
    executor : ThreadPoolExecutor
        Executor used to run concurrent download tasks.
    config : Config
        Project configuration object providing headers, paths and worker settings.
    """
    page_id = 1
    headers = config.data.headers
    raw_data_path = Path(config.paths.raw_data)

    with tqdm(desc=f"Downloading games for {player_name}", unit="game") as pbar:
        while True:
            url = f"https://www.chessgames.com/perl/chess.pl?page={page_id}&pid={player_id}"
            tree = fetch_url(url, headers)
            table = tree.xpath("//table[@cellpadding='3']")

            if not table:
                break

            rows = table[0].xpath(".//tr")[1:]
            futures = []

            for row in rows:
                cells = row.xpath(".//td")
                link = cells[0].xpath(".//a/@href")
                gid = link[0].split("gid=")[-1]
                download_url = (
                    f"https://www.chessgames.com/njs/api/game/downloadPGN/{gid}"
                )

                futures.append(
                    executor.submit(
                        download_pgn,
                        player_id,
                        gid,
                        download_url,
                        raw_data_path,
                        headers,
                    )
                )

            for future in as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    logger.warning(f"An error occurred while downloading a game: {e}")

            page_id += 1


def fetch_all_games(config: Config) -> None:
    """Coordinate the retrieval of games for all players specified in the configuration.

    This function initializes a thread pool according to the configured maximum worker
    count and invokes `fetch_chessgames` for each configured player.
    """
    logger.info("Commencing retrieval of all games...")

    with ThreadPoolExecutor(max_workers=config.data.max_workers) as executor:
        for player_id, player_name in config.data.players.items():
            fetch_chessgames(player_id, player_name, executor, config)
