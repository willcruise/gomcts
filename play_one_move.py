import argparse
import torch
from board import Board9x9, Board
from mcts import ScoreAwareMCTS
from policyneural import MLPPolicyValue, infer_policy_value


def main(size: int = 9, sims: int = 50, temp: float = 1.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MLPPolicyValue(device=device)
    print("Using device:", device)

    # Wire MCTS to use our policy/value net via inference wrapper
    def legal_actions_fn(board):
        return board.legal_moves()

    def next_state_fn(board, action):
        b = board.clone()
        b.play(action)
        return b

    def is_terminal_fn(board):
        return board.is_terminal()

    def current_player_fn(board):
        # Map +1 black to 0, -1 white to 1 to match two-player toggle if needed
        return 0 if board.turn == 1 else 1

    def policy_value_fn(board):
        return infer_policy_value(net, board)

    # Create board, then adapt action space to this board's size
    board = Board(size)
    num_actions = getattr(board, 'size', size) ** 2 + 1
    mcts = ScoreAwareMCTS(
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
    mcts.run(board, num_simulations=int(sims))
    a = mcts.choose_action(temp=float(temp))
    pi = mcts.get_action_probs(temp=float(temp))
    print("chosen action:", a, "(PASS)" if a == getattr(board, 'pass_index', getattr(board, 'size', 9) ** 2) else "")
    print("policy pi sum:", float(pi.sum()), "nonzeros:", int((pi > 0).sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play one move demo")
    parser.add_argument("--size", type=int, default=9, help="board size N")
    parser.add_argument("--sims", type=int, default=50, help="MCTS simulations")
    parser.add_argument("--temp", type=float, default=1.0, help="temperature for action selection")
    args = parser.parse_args()
    main(size=args.size, sims=args.sims, temp=args.temp)

