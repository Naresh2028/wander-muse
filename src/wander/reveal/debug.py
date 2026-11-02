"""Make the invisible visible: a contact sheet of the layers SAM + our code produced.

Each tile shows one layer over a checkerboard, so transparent areas are obvious. If a layer looks
wrong here (a sky that swallowed the mountains, a boat with a ragged edge) the parallax will look
wrong too, and this is the fastest place to see why. It is also a nice picture for the demo video.
"""

from __future__ import annotations

import cv2
import numpy as np

from .layers import Layer


def _checkerboard(h: int, w: int, cell: int = 16) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    tile = (((yy // cell) + (xx // cell)) % 2).astype(np.uint8)
    board = np.where(tile[..., None] == 0, 200, 150).astype(np.uint8)
    return np.repeat(board, 3, axis=2) if board.shape[2] == 1 else board


def layers_contact_sheet(layers: list[Layer], tile_width: int = 384, columns: int = 3) -> np.ndarray:
    tiles = []
    for layer in layers:
        h, w = layer.rgb.shape[:2]
        th = round(tile_width * h / w)
        rgb = cv2.resize(layer.rgb, (tile_width, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        a = cv2.resize(layer.alpha, (tile_width, th), interpolation=cv2.INTER_AREA)[..., None]
        board = _checkerboard(th, tile_width).astype(np.float32)
        tile = np.clip(rgb * a + board * (1 - a), 0, 255).astype(np.uint8)
        label = f"{layer.name}  (depth {layer.depth:.2f})"
        cv2.rectangle(tile, (0, 0), (tile_width, 24), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)
