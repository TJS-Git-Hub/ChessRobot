import unittest
import os
from unittest.mock import Mock
from pathlib import Path

from game.ai import ChessAI, DEFAULT_ENGINE, board_to_grid, uci_to_game
from game.board import ChessBoard
from game.uci_engine import EngineError
from game.xiangqi_rules import START_FEN, to_fen

ENGINE = Path(os.environ.get('PIKAFISH_TEST_ENGINE', DEFAULT_ENGINE))


class AdapterTests(unittest.TestCase):
    def test_opening_fen_and_coordinates(self):
        board = ChessBoard()
        self.assertEqual(to_fen(board_to_grid(board), board.red_turn), START_FEN)
        self.assertEqual(uci_to_game('a3a4'), '0605')
        self.assertEqual(uci_to_game('i9i8'), '8081')

    def test_bad_and_illegal_move_are_rejected(self):
        board = ChessBoard()
        board.make_move('0605')
        ai = ChessAI(board)
        ai.engine = Mock()
        for value in ('a0a1', 'bad', '(none)', None):
            with self.subTest(value=value):
                ai.engine.bestmove.return_value = value
                with self.assertRaises(EngineError):
                    ai.get_best_move()
        ai.close()

    @unittest.skipUnless(ENGINE.is_file(), 'Pikafish runtime asset not installed')
    def test_real_engine_after_human_move(self):
        board = ChessBoard()
        board.make_move('0605')
        ai = ChessAI(board, engine_path=ENGINE, movetime_ms=300)
        try:
            move = ai.get_best_move()
            self.assertTrue(board.is_move_valid(move), move)
            board.make_move(move)
            self.assertTrue(board.red_turn)
        finally:
            ai.close()

    def test_invalid_pieces_fail_closed(self):
        board = ChessBoard()
        board.board[0][0] = ('black', 'unknown')
        with self.assertRaises(ValueError):
            board_to_grid(board)

    def test_move_exposing_king_is_rejected(self):
        board = ChessBoard()
        board.board = [[None] * 9 for _ in range(10)]
        board.board[0][4] = ('black', 'jiang')
        board.board[1][4] = ('black', 'ju')
        board.board[8][4] = ('red', 'ju')
        board.board[9][3] = ('red', 'shuai')
        board.red_turn = False
        self.assertTrue(board.is_move_valid('4151'))
        ai = ChessAI(board)
        ai.engine = Mock()
        ai.engine.bestmove.return_value = 'e8f8'
        with self.assertRaises(EngineError):
            ai.get_best_move()
        ai.close()

if __name__ == '__main__':
    unittest.main()
