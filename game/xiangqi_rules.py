"""Xiangqi board and legal move generation. Rows 0..9 top (black), columns 0..8."""
from dataclasses import dataclass

START_FEN = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'
PIECES = {'r':'车','n':'马','b':'象','a':'士','k':'将','c':'炮','p':'卒',
          'R':'车','N':'马','B':'相','A':'仕','K':'帅','C':'炮','P':'兵'}
FILES='abcdefghi'

def parse_fen(fen):
    cells=[]
    for rank in fen.split()[0].split('/'):
        row=[]
        for char in rank:
            if char.isdigit(): row += ['.'] * int(char)
            else: row.append(char)
        if len(row)!=9: raise ValueError('Invalid FEN row')
        cells.append(row)
    if len(cells)!=10: raise ValueError('Invalid FEN')
    return cells

def to_fen(board, red=True):
    ranks=[]
    for row in board:
        rank=''; count=0
        for piece in row:
            if piece=='.': count+=1
            else:
                if count: rank+=str(count); count=0
                rank+=piece
        if count:rank+=str(count)
        ranks.append(rank)
    return '/'.join(ranks)+(' w - - 0 1' if red else ' b - - 0 1')

def red_piece(p): return p!='.' and p.isupper()
def same_color(a,b): return b!='.' and a.isupper()==b.isupper()
def in_board(r,c): return 0<=r<10 and 0<=c<9

def pseudo_moves(board,r,c):
    p=board[r][c]
    if p=='.':return
    red=p.isupper(); kind=p.lower()
    def can(rr,cc):return in_board(rr,cc) and not same_color(p,board[rr][cc])
    if kind=='r' or kind=='c':
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            rr,cc=r+dr,c+dc; jumped=False
            while in_board(rr,cc):
                q=board[rr][cc]
                if kind=='r':
                    if q=='.':yield (rr,cc)
                    else:
                        if not same_color(p,q):yield (rr,cc)
                        break
                elif not jumped:
                    if q=='.':yield (rr,cc)
                    else:jumped=True
                else:
                    if q!='.':
                        if not same_color(p,q):yield (rr,cc)
                        break
                rr+=dr;cc+=dc
    elif kind=='n':
        for leg_r,leg_c,outs in [(-1,0,[(-2,-1),(-2,1)]),(1,0,[(2,-1),(2,1)]),
                                 (0,-1,[(-1,-2),(1,-2)]),(0,1,[(-1,2),(1,2)])]:
            lr,lc=r+leg_r,c+leg_c
            if in_board(lr,lc) and board[lr][lc]=='.':
                for dr,dc in outs:
                    if can(r+dr,c+dc):yield(r+dr,c+dc)
    elif kind=='b':
        for dr in (-2,2):
            for dc in (-2,2):
                rr,cc=r+dr,c+dc
                if in_board(rr,cc) and ((rr>=5) if red else (rr<=4)) and board[r+dr//2][c+dc//2]=='.' and can(rr,cc):yield(rr,cc)
    elif kind=='a':
        for dr in (-1,1):
            for dc in (-1,1):
                rr,cc=r+dr,c+dc
                if 3<=cc<=5 and ((7<=rr<=9) if red else (0<=rr<=2)) and can(rr,cc):yield(rr,cc)
    elif kind=='k':
        for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            rr,cc=r+dr,c+dc
            if 3<=cc<=5 and ((7<=rr<=9) if red else (0<=rr<=2)) and can(rr,cc):yield(rr,cc)
        # Flying general: allow capture across open file.
        dr=-1 if red else 1;rr=r+dr
        while in_board(rr,c):
            if board[rr][c]!='.':
                if board[rr][c].lower()=='k' and not same_color(p,board[rr][c]):yield(rr,c)
                break
            rr+=dr
    elif kind=='p':
        dr=-1 if red else 1
        if can(r+dr,c):yield(r+dr,c)
        if (r<=4 if red else r>=5):
            for dc in (-1,1):
                if can(r,c+dc):yield(r,c+dc)

def make(board,move):
    (r,c),(rr,cc)=move
    out=[row.copy() for row in board]
    out[rr][cc]=out[r][c];out[r][c]='.'
    return out

def attacked(board, r, c, by_red):
    # Fully general attack map, including king flying capture.
    for sr in range(10):
        for sc in range(9):
            p=board[sr][sc]
            if p!='.' and p.isupper()==by_red and (r,c) in pseudo_moves(board,sr,sc):return True
    return False

def in_check(board,red):
    king='K' if red else 'k'
    for r,row in enumerate(board):
        for c,p in enumerate(row):
            if p==king:return attacked(board,r,c,not red)
    return True

def legal_moves(board,red):
    for r in range(10):
        for c in range(9):
            p=board[r][c]
            if p!='.' and p.isupper()==red:
                for dest in pseudo_moves(board,r,c):
                    move=((r,c),dest)
                    if not in_check(make(board,move),red):yield move

def to_uci(move):
    (r,c),(rr,cc)=move
    return FILES[c]+str(9-r)+FILES[cc]+str(9-rr)

def from_uci(s):
    if len(s)!=4 or s[0] not in FILES or s[2] not in FILES or s[1] not in '0123456789' or s[3] not in '0123456789':raise ValueError('invalid UCI move: '+s)
    return ((9-int(s[1]),FILES.index(s[0])),(9-int(s[3]),FILES.index(s[2])))
