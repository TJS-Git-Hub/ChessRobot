"""Pikafish decision adapter for the existing ChessBoard interface."""
import os
from pathlib import Path

from game.uci_engine import UCIEngine, EngineError
from game.xiangqi_rules import from_uci, legal_moves, to_fen


PIECES = {
    'ju': 'r', 'ma': 'n', 'xiang': 'b', 'shi': 'a',
    'jiang': 'k', 'shuai': 'k', 'pao': 'c', 'zu': 'p', 'bing': 'p',
}
DEFAULT_ENGINE = Path(__file__).resolve().parent / 'pikafish' / 'Pikafish-Windows-x86-64-universal.exe'


def board_to_grid(board):
    """Convert board[y][x] to Pikafish's 10x9 character grid."""
    if len(board.board) != 10 or any(len(row) != 9 for row in board.board):
        raise ValueError('棋盘尺寸不是 10×9')
    grid = []
    for row in board.board:
        result = []
        for piece in row:
            if piece is None:
                result.append('.')
                continue
            color, kind = piece
            if color not in ('red', 'black') or kind not in PIECES:
                raise ValueError(f'未知棋子: {piece!r}')
            char = PIECES[kind]
            result.append(char.upper() if color == 'red' else char)
        grid.append(result)
    if sum(row.count('K') for row in grid) != 1 or sum(row.count('k') for row in grid) != 1:
        raise ValueError('局面必须各有一个红帅和黑将')
    return grid


def uci_to_game(move):
    (y1, x1), (y2, x2) = from_uci(move)
    return f'{x1}{y1}{x2}{y2}'


class ChessAI:
    def __init__(self, board, engine_path=None, movetime_ms=3000):
        self.board = board
        self.engine_path = Path(engine_path) if engine_path else DEFAULT_ENGINE
        self.movetime_ms = int(movetime_ms)
        if self.movetime_ms < 100:
            raise ValueError('Pikafish 思考时间不能小于 100 毫秒')
        self.engine = None

    def get_best_move(self):
        """Return the existing four-digit move format, or '' if no legal move."""
        grid = board_to_grid(self.board)
        side = self.board.red_turn
        legal = set(legal_moves(grid, side))
        if not legal:
            return ''
        if self.engine is None:
            if not self.engine_path.is_file():
                raise EngineError(f'未找到 Pikafish 引擎: {self.engine_path}')
            if not (self.engine_path.parent / 'pikafish.nnue').is_file():
                raise EngineError(f'未找到引擎权重: {self.engine_path.parent / "pikafish.nnue"}')
            self.engine = UCIEngine(str(self.engine_path), threads=max(1, min(8, os.cpu_count() or 2)))
        move = self.engine.bestmove(to_fen(grid, side), self.movetime_ms)
        try:
            parsed = from_uci(move)
        except (TypeError, ValueError) as exc:
            raise EngineError(f'Pikafish 返回无效着法: {move!r}') from exc
        result = uci_to_game(move)
        if parsed not in legal or not self.board.is_move_valid(result):
            raise EngineError(f'Pikafish 着法未通过双重合法性检查: {move} / {result}')
        print(f'🤖 Pikafish 决策: {move} → {result}')
        return result

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None
