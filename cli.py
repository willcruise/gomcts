import sys
from board import Board9x9, Board
from mcts import ScoreAwareMCTS
from policyneural import MLPPolicyValue, infer_policy_value
from selftraining import evaluate_winner_simple


def _build_mcts(net: MLPPolicyValue, size: int) -> ScoreAwareMCTS:
    def legal_actions_fn(b: Board9x9):
        return b.legal_moves()

    def next_state_fn(b: Board9x9, action: int):
        nb = b.clone()
        nb.play(action)
        return nb

    def is_terminal_fn(b: Board9x9):
        return b.is_terminal()

    def current_player_fn(b: Board9x9):
        return 0 if b.turn == 1 else 1

    def policy_value_fn(b: Board9x9):
        return infer_policy_value(net, b)

    num_actions = size * size + 1
    return ScoreAwareMCTS(
        num_actions=num_actions,
        legal_actions_fn=legal_actions_fn,
        next_state_fn=next_state_fn,
        is_terminal_fn=is_terminal_fn,
        policy_value_fn=policy_value_fn,
        current_player_fn=current_player_fn,
        c_puct=2.0,
        root_dirichlet_alpha=None,
        root_dirichlet_frac=0.25,
        root_dirichlet_c0=10.0,
    )


def _print_board(b: Board9x9) -> None:
    print("Turn:", "B" if b.turn == 1 else "W")
    # Build column labels using Go convention (skip 'I')
    N = getattr(b, 'size', 9)
    alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # 'I' skipped
    letters = [alphabet[i] for i in range(min(N, len(alphabet)))]
    cols = " ".join(letters)
    print("   " + cols)
    for row_num in range(N, 0, -1):
        r = N - row_num
        row_syms = []
        for c in range(N):
            v = b.grid[r, c]
            if v > 0:
                row_syms.append("X")
            elif v < 0:
                row_syms.append("O")
            else:
                row_syms.append(".")
        print(f"{row_num:>2} " + " ".join(row_syms))


def _index_to_rc(b: Board9x9, a: int) -> tuple:
    if a == getattr(b, 'pass_index', b.size * b.size):
        return (None, None)
    N = getattr(b, 'size', 9)
    r, c = divmod(int(a), N)
    return (r, c)


def _parse_coord(s: str, size: int) -> int:
    if s is None:
        raise ValueError("empty coord")
    s = s.strip().upper()
    if s == "PASS":
        return size * size
    if len(s) < 2:
        raise ValueError("coord too short")
    col_letter = s[0]
    row_part = s[1:]

    # Build dynamic letter mapping using Go convention (skip 'I')
    alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # 'I' skipped
    if size > len(alphabet):
        raise ValueError("size too large for coordinate mapping")
    letters = [alphabet[i] for i in range(size)]
    if col_letter not in letters:
        raise ValueError("invalid column letter")
    col = letters.index(col_letter)

    try:
        row_num = int(row_part)
    except Exception:
        raise ValueError("invalid row number")
    if not (1 <= row_num <= size):
        raise ValueError("row out of range")

    r = size - row_num
    c = col
    return r * size + c


def cli() -> None:
    net = MLPPolicyValue()
    board = Board(9, enforce_rules=True, forbid_suicide=True, ko_rule='simple')

    print("gomcts CLI ready. Commands: boardsize N | clearboard | showboard | genmove [b|w] [sims] [temp] | play [b|w] [coord|pass] | finalscore | undo | exit")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "exit":
            break
        elif cmd in ("clearboard", "clear_board"):
            # Keep current size on clear
            board = Board(getattr(board, 'size', 9), enforce_rules=True, forbid_suicide=True, ko_rule='simple')
            print("= cleared")
        elif cmd in ("showboard", "show_board"):
            _print_board(board)
        elif cmd == "boardsize":
            if len(parts) < 2:
                print("? usage: boardsize N")
                continue
            try:
                N = int(parts[1])
                if N < 2 or N > 25:
                    raise ValueError()
            except Exception:
                print("? invalid size (2..25)")
                continue
            board = Board(N, enforce_rules=True, forbid_suicide=True, ko_rule='simple')
            print(f"= size set to {N}")
        elif cmd == "genmove":
            color = None
            sims = 50
            temp = 1.0
            if len(parts) >= 2:
                color = parts[1].lower()
            if len(parts) >= 3:
                try:
                    sims = int(parts[2])
                except ValueError:
                    pass
            if len(parts) >= 4:
                try:
                    temp = float(parts[3])
                except ValueError:
                    pass

            if color in ("b", "black"):
                board.turn = 1
            elif color in ("w", "white"):
                board.turn = -1

            mcts = _build_mcts(net, size=getattr(board, 'size', 9))
            mcts.run(board, num_simulations=sims)
            action = mcts.choose_action(temp=temp)
            r, c = _index_to_rc(board, action)
            board.play(action)
            if action == getattr(board, 'pass_index', board.size * board.size):
                print("= pass")
            else:
                print(f"= move {action} (r={r}, c={c})")
            _print_board(board)
        elif cmd == "play":
            if len(parts) < 3:
                print("? usage: play [b|w] [coord|pass]")
                continue
            color = parts[1].lower()
            move_str = parts[2]
            if color in ("b", "black"):
                board.turn = 1
            elif color in ("w", "white"):
                board.turn = -1
            else:
                print("? invalid color (use b|w)")
                continue

            try:
                action = _parse_coord(move_str, getattr(board, 'size', 9))
            except Exception:
                print("? invalid coordinate (e.g., A7 or PASS)")
                continue

            before_len = len(getattr(board, 'history', []))
            board.play(action)
            after_len = len(getattr(board, 'history', []))
            if after_len == before_len:
                print("? illegal move (blocked/suicide/ko/occupied)")
                _print_board(board)
                continue
            if action == getattr(board, 'pass_index', board.size * board.size):
                print("= pass")
            else:
                r, c = _index_to_rc(board, action)
                print(f"= played {move_str.upper()} (r={r}, c={c})")
            _print_board(board)
        elif cmd == "finalscore":
            black = int((board.grid == 1).sum())
            white = int((board.grid == -1).sum())
            score = float(black - white)
            winner = evaluate_winner_simple(board, komi=0.0)
            if winner > 0:
                print(f"= B+{abs(score):.1f} (stones B={black}, W={white})")
            elif winner < 0:
                print(f"= W+{abs(score):.1f} (stones B={black}, W={white})")
            else:
                print(f"= Draw {score:.1f} (stones B={black}, W={white})")
        elif cmd == "undo":
            if board.undo():
                print("= undone")
                _print_board(board)
            else:
                print("? nothing to undo")
        else:
            print("? unknown command")


if __name__ == "__main__":
    cli()


