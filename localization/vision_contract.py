"""Hardware-free board contract. Rows increase from black to red, a..i columns.

Tensor channels intentionally follow ChessBoard.get_state_tensor(), preserving
the existing AI input order. FEN uses red 'w'/black 'b' as side to move.
"""
import numpy as np

CHANNEL_ORDER = "krncbapKRNCBAP"
KINDS = dict(zip("krncbapKRNCBAP", ("jiang","ju","ma","pao","xiang","shi","zu",
                                    "shuai","ju","ma","pao","xiang","shi","bing")))


def validate_grid(grid, allow_unknown=False):
    if not isinstance(grid,(list,tuple)) or len(grid) != 10 or any(len(row)!=9 for row in grid):
        raise ValueError("Board must have 10 rows and 9 columns")
    allowed = set(CHANNEL_ORDER+"."+("x" if allow_unknown else ""))
    if any(not isinstance(ch,str) or ch not in allowed for row in grid for ch in row):
        raise ValueError("Unknown or invalid board cell")
    return [list(row) for row in grid]


def parse_coordinate(pos):
    if not isinstance(pos,str) or len(pos)!=2 or pos[0].lower() not in "abcdefghi" or pos[1] not in "0123456789":
        raise ValueError("Coordinate must be a0..i9")
    return ord(pos[0].lower())-97,int(pos[1])


def coordinate(x,y):
    if type(x) is not int or type(y) is not int or not (0<=x<9 and 0<=y<10):
        raise ValueError("Coordinate out of board")
    return f"{chr(97+x)}{y}"


def grid_to_tensor(grid):
    grid=validate_grid(grid)
    result=np.zeros((14,10,9),np.float32)
    for r,row in enumerate(grid):
        for c,ch in enumerate(row):
            if ch!='.':result[CHANNEL_ORDER.index(ch),r,c]=1
    return result


def grid_to_board(grid):
    return [[None if ch=='.' else ("red" if ch.isupper() else "black",KINDS[ch])
             for ch in row] for row in validate_grid(grid)]


def board_to_grid(board):
    if len(board)!=10 or any(len(row)!=9 for row in board):raise ValueError("Invalid game board")
    inverse={("red" if ch.isupper() else "black",kind):ch for ch,kind in KINDS.items()}
    return [["." if value is None else inverse[tuple(value)] for value in row] for row in board]


def grid_to_fen(grid, red_turn=True):
    rows=[]
    for row in validate_grid(grid):
        text="";empty=0
        for ch in row:
            if ch=='.':empty+=1
            else:
                if empty:text+=str(empty);empty=0
                text+=ch
        if empty:text+=str(empty)
        rows.append(text)
    return "/".join(rows)+(" w" if red_turn else " b")+" - - 0 1"


def fen_to_grid(fen):
    parts=fen.split()
    if len(parts)<2 or parts[1] not in ('w','b'):raise ValueError("FEN needs explicit side to move")
    grid=[]
    for rank in parts[0].split('/'):
        row=[]
        for ch in rank:
            if ch in '123456789':row+=['.']*int(ch)
            else:row.append(ch)
        grid.append(row)
    return validate_grid(grid),parts[1]=='w'


def moved_grid(grid,source,target):
    result=validate_grid(grid);x1,y1=parse_coordinate(source);x2,y2=parse_coordinate(target)
    piece=result[y1][x1]
    if piece=='.' or source.lower()==target.lower():raise ValueError("Invalid move source")
    if result[y2][x2]!='.' and result[y2][x2].isupper()==piece.isupper():raise ValueError("Own piece at target")
    result[y1][x1]='.';result[y2][x2]=piece
    return result


def infer_move(before,after):
    before,after=validate_grid(before),validate_grid(after)
    changes=[(r,c) for r in range(10) for c in range(9) if before[r][c]!=after[r][c]]
    if len(changes)!=2:return None
    lost=[(r,c) for r,c in changes if before[r][c]!='.' and after[r][c]=='.']
    if len(lost)!=1:return None
    r1,c1=lost[0];r2,c2=next(p for p in changes if p!=lost[0])
    source,target=coordinate(c1,r1),coordinate(c2,r2)
    try:
        if moved_grid(before,source,target)!=after:return None
    except ValueError:return None
    return source,target,before[r1][c1],before[r2][c2]!='.'
