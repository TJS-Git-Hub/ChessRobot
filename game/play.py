from game.board import ChessBoard
from game.ai import ChessAI


def main():
    board = ChessBoard()
    ai = ChessAI(board)

    print("=" * 30)
    print("      中国象棋 - 人类 VS 深度学习 AI")
    print("=" * 30)
    print("👉 输入走法示例：'6947' (表示坐标 x=6,y=9 的棋子移动到 x=4,y=7)")
    print("👉 输入 'q' 退出游戏")
    print("红方（人类）先手！\n")

    while True:
        board.print_board()

        # 1. 检查游戏是否结束
        result = board.check_game_over()
        if result:
            print(f"\n🏆 {result}")
            break

        # 2. 回合判定
        if board.red_turn:
            print("--- 【轮到你走棋了 (红方)】 ---")
            valid_moves = board.get_all_legal_moves(is_red=True)
            if not valid_moves:
                print("🏆 红方无路可走，黑方胜利！")
                break

            human_move = input("请输入你的走法: ").strip()

            if human_move.lower() == 'q':
                print("游戏退出。")
                break

            if human_move not in valid_moves:
                print("❌ 非法走法！请检查坐标或是否违反象棋规则。")
                continue

            board.make_move(human_move)
        else:
            print("\n--- 【AI 正在思考 (黑方)】 ---")
            ai_move = ai.get_best_move()
            if not ai_move:
                print("🏆 黑方无路可走，红方胜利！")
                break

            board.make_move(ai_move)


if __name__ == "__main__":
    main()