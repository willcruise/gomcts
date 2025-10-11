from copy import deepcopy
from typing import Optional
import numpy as np
import rules


class Board:
    def __init__(self, size: int = 9, strict_illegals: bool = False,
                 enforce_rules: bool = False, forbid_suicide: bool = False,
                 ko_rule: Optional[str] = None):
        # board size (N x N)
        self.size = int(size)
        # +1 black, -1 white
        self.turn = 1
        # 0 empty, +1 black stone, -1 white stone
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)
        # Minimal ruleset without captures/ko.
        self.history = []  # list of moves (0..(N*N-1)) or pass_index
        # Rule information channel value in [0,1].
        self.rule_info = 0.0
        # Optional: raise on illegal overwrites if enabled (default off to preserve behavior)
        self._strict_illegals = bool(strict_illegals)
        # Komi used for terminal result evaluation
        self.komi = 6.5
        # Optional enhanced rules (backwards-compatible: disabled by default)
        self._enforce_rules = bool(enforce_rules)
        self._forbid_suicide = bool(forbid_suicide)
        # ko_rule: None (disabled), 'simple' (no immediate repetition), 'psk' (positional superko)
        if ko_rule not in (None, 'simple', 'psk'):
            raise ValueError("ko_rule must be one of None, 'simple', or 'psk'")
        self._ko_rule = ko_rule
        # Internals for undo and ko tracking
        self._move_stack = []  # list of dicts with move, placed_color, captures
        self._position_history = [self._hash_from(self.grid, self.turn)]
        # Turn-agnostic history for positional superko
        self._position_history_pos = [self._poshash_from(self.grid)]
        self._psk_set = {self._position_history_pos[0]}
        # Track total captures by each player (for scoring)
        self.captures_black = 0
        self.captures_white = 0
        # Zobrist hashing tables and incremental hash
        self._zobrist_init()
        self._zobrist_hash = self._zobrist_hash_full()
        # Cache for legal moves keyed by position hash/config
        self._legal_cache_key = None
        self._legal_cache = None

    @property
    def pass_index(self) -> int:
        return self.size * self.size

    def clone(self):
        """Efficient clone: copy arrays/lists without Python deepcopy overhead."""
        b = Board(self.size,
                  strict_illegals=self._strict_illegals,
                  enforce_rules=self._enforce_rules,
                  forbid_suicide=self._forbid_suicide,
                  ko_rule=self._ko_rule)
        b.turn = int(self.turn)
        b.grid = self.grid.copy()
        b.history = list(self.history)
        b.rule_info = float(self.rule_info)
        b.komi = float(self.komi)
        # Copy internals (lightweight):
        # - position histories are small bytes arrays; copy lists directly
        b._position_history = list(self._position_history)
        if hasattr(self, '_position_history_pos'):
            b._position_history_pos = list(self._position_history_pos)
        if hasattr(self, '_psk_set'):
            b._psk_set = set(self._psk_set)
        # - move stack is not needed for forward-only simulation; avoid deep copy
        b._move_stack = []
        b.captures_black = int(self.captures_black)
        b.captures_white = int(self.captures_white)
        # Do not copy caches
        return b

    def legal_moves(self):
        """
        Return np.array of legal move indices in [0..N*N] (inclusive of pass).
        Disallow placing on occupied intersections; PASS always legal.
        If enhanced rules enabled, also disallow suicide and ko per configuration.
        """
        # Cache lookup by turn-aware position hash + rule flags
        try:
            key = (self._position_history[-1], self._enforce_rules, self._forbid_suicide, self._ko_rule)
            if self._legal_cache_key == key and self._legal_cache is not None:
                return self._legal_cache.copy()
        except Exception:
            pass
        if not self._enforce_rules and not self._forbid_suicide and self._ko_rule is None:
            legal = []
            N = self.size
            for r in range(N):
                for c in range(N):
                    if self.grid[r, c] == 0:
                        legal.append(r * N + c)
            legal.append(self.pass_index)
            arr = np.array(legal, dtype=np.int64)
            try:
                self._legal_cache_key = key
                self._legal_cache = arr
            except Exception:
                pass
            return arr

        legal = []
        N = self.size
        empties = np.flatnonzero(self.grid.reshape(-1) == 0)
        for move in empties.tolist():
            if self._is_legal(int(move)):
                legal.append(int(move))
        legal.append(self.pass_index)
        arr = np.array(legal, dtype=np.int64)
        try:
            self._legal_cache_key = key
            self._legal_cache = arr
        except Exception:
            pass
        return arr

    def play(self, move: int):
        """Apply move in-place. If rules are enabled, enforce captures/suicide/ko.

        Behavior with defaults remains identical to the previous implementation.
        """
        # Invalidate legal cache on any state change
        self._legal_cache_key = None; self._legal_cache = None
        if move == self.pass_index:
            self.history.append(self.pass_index)
            # Update position history for pass (turn changes, stones unchanged)
            self.turn *= -1
            # Update Zobrist for turn flip
            try:
                self._zobrist_hash ^= self._zobrist_turn_key
            except Exception:
                pass
            new_hash = self._hash_from(self.grid, self.turn)
            self._position_history.append(new_hash)
            new_poshash = self._poshash_from(self.grid)
            self._position_history_pos.append(new_poshash)
            if self._ko_rule == 'psk':
                self._psk_set.add(new_poshash)
            # Record a lightweight stack frame for undo
            self._move_stack.append({
                'move': self.pass_index,
                'placed_color': 0,
                'captures': [],
            })
            return
        if self._enforce_rules or self._forbid_suicide or self._ko_rule is not None:
            applied = self._apply_move_with_rules(move)
            if not applied:
                if self._strict_illegals:
                    raise ValueError("Illegal move under rules configuration")
                # ignore illegal move to preserve workflow
                return
            return
        # Legacy behavior (no captures/ko)
        N = self.size
        r, c = divmod(int(move), N)
        if self.grid[r, c] != 0:
            if self._strict_illegals:
                raise ValueError("Illegal move: point already occupied")
            # ignore illegal overwrite to preserve previous workflow
            return
        color = 1 if self.turn == 1 else -1
        self.grid[r, c] = color
        # Update Zobrist for placement
        try:
            self._zobrist_hash ^= (self._zobrist_black[r, c] if color == 1 else self._zobrist_white[r, c])
        except Exception:
            pass
        self.history.append(int(move))
        self.turn *= -1
        try:
            self._zobrist_hash ^= self._zobrist_turn_key
        except Exception:
            pass
        # Update position history even in legacy mode to keep internals consistent
        new_hash = self._hash_from(self.grid, self.turn)
        self._position_history.append(new_hash)
        new_poshash = self._poshash_from(self.grid)
        self._position_history_pos.append(new_poshash)
        if self._ko_rule == 'psk':
            self._psk_set.add(new_poshash)
        self._move_stack.append({
            'move': int(move),
            'placed_color': 1 if self.turn == -1 else -1,  # color that just played
            'captures': [],
        })

    def is_terminal(self) -> bool:
        # Terminal when two consecutive passes
        return (
            len(self.history) >= 2
            and self.history[-1] == self.pass_index
            and self.history[-2] == self.pass_index
        )

    def result(self) -> float:
        """Final result from BLACK's perspective in [-1,1].

        Uses capture-aware proxy: score = (B - W) + (capsB - capsW).
        Returns +1 for Black win, -1 for White win, 0 for draw.
        """
        score = rules.capture_aware_score(self.grid, self.captures_black, self.captures_white)
        if score > 0:
            return 1.0
        if score < 0:
            return -1.0
        return 0.0

    def to_features(self) -> np.ndarray:
        """
        Return flattened feature planes as float32.
        Planes:
          0: black stones
          1: white stones
          2: rule information (constant plane filled with rule_info)
          3: to-play (all 1.0 if black to move, else 0.0)
        Total: 4 x N x N features.
        """
        N = self.size
        planes = np.zeros((4, N, N), dtype=np.float32)
        planes[0][self.grid == 1] = 1.0
        planes[1][self.grid == -1] = 1.0
        planes[2, :, :] = float(self.rule_info)
        planes[3, :, :] = 1.0 if self.turn == 1 else 0.0
        return planes.reshape(-1)

    def undo(self) -> bool:
        """Undo last move. Returns True if undone, False if no history."""
        self._legal_cache_key = None; self._legal_cache = None
        if not self.history:
            return False
        last = self.history.pop()
        # Revert turn first (since play() flipped it after move)
        self.turn *= -1
        # Revert internals
        if self._position_history:
            self._position_history.pop()
        if hasattr(self, '_position_history_pos') and self._position_history_pos:
            self._position_history_pos.pop()
        if self._ko_rule == 'psk':
            # Recompute PSK set from the current linear history to avoid retaining
            # states that belonged only to undone branches.
            self._psk_set = set(self._position_history_pos)
        if last != self.pass_index:
            N = self.size
            r, c = divmod(int(last), N)
            # Clear the placed stone
            color_cleared = self.grid[r, c]
            self.grid[r, c] = 0
            # Update Zobrist for removal (XOR same key)
            try:
                if int(color_cleared) == 1:
                    self._zobrist_hash ^= self._zobrist_black[r, c]
                elif int(color_cleared) == -1:
                    self._zobrist_hash ^= self._zobrist_white[r, c]
            except Exception:
                pass
        # Restore captures if any
        if self._move_stack:
            info = self._move_stack.pop()
            # Roll back capture counters corresponding to the move being undone
            if info.get('captures'):
                placed_color = int(info.get('placed_color', 0))
                num_caps = int(len(info['captures']))
                if placed_color == 1:
                    self.captures_black -= num_caps
                elif placed_color == -1:
                    self.captures_white -= num_caps
            for rr, cc, color in info.get('captures', []):
                self.grid[rr, cc] = color
                # Reapply captured stones into Zobrist
                try:
                    if int(color) == 1:
                        self._zobrist_hash ^= self._zobrist_black[rr, cc]
                    elif int(color) == -1:
                        self._zobrist_hash ^= self._zobrist_white[rr, cc]
                except Exception:
                    pass
        return True

    # -------------------- Enhanced rules helpers --------------------

    def _hash_from(self, grid: np.ndarray, turn: int) -> bytes:
        # Include side-to-play; prefer Zobrist hash when available for speed
        try:
            z = self._zobrist_hash if turn == self.turn else self._zobrist_hash ^ self._zobrist_turn_key
            # Return as bytes (8-byte little-endian)
            return int(z & 0xFFFFFFFFFFFFFFFF).to_bytes(8, byteorder="little", signed=False)
        except Exception:
            return (b"B" if turn == 1 else b"W") + grid.tobytes()

    def _poshash_from(self, grid: np.ndarray) -> bytes:
        try:
            return int(self._zobrist_hash & 0xFFFFFFFFFFFFFFFF).to_bytes(8, byteorder="little", signed=False)
        except Exception:
            return grid.tobytes()

    # -------------------- Zobrist hashing helpers --------------------
    def _zobrist_init(self) -> None:
        rng = np.random.RandomState(123456789)
        N = int(self.size)
        # 3 states per point: empty, black, white -> we use only black/white keys
        self._zobrist_black = rng.randint(1, 2**63 - 1, size=(N, N), dtype=np.int64)
        self._zobrist_white = rng.randint(1, 2**63 - 1, size=(N, N), dtype=np.int64)
        self._zobrist_turn_key = np.int64(rng.randint(1, 2**63 - 1, dtype=np.int64))

    def _zobrist_hash_full(self) -> np.int64:
        N = int(self.size)
        h = np.int64(0)
        for r in range(N):
            for c in range(N):
                v = int(self.grid[r, c])
                if v == 1:
                    h ^= self._zobrist_black[r, c]
                elif v == -1:
                    h ^= self._zobrist_white[r, c]
        if int(self.turn) == -1:
            h ^= self._zobrist_turn_key
        return np.int64(h)

    def _is_legal(self, move: int) -> bool:
        if move == self.pass_index:
            return True
        N = self.size
        r, c = divmod(int(move), N)
        if self.grid[r, c] != 0:
            return False
        color = 1 if self.turn == 1 else -1
        # Simulate placement and captures using shared rules helpers
        sim_grid, _captures, captured_any = rules.simulate_place_and_capture(self.grid, r, c, color)

        # Suicide check (forbid only if configured)
        if self._forbid_suicide:
            if rules.is_suicide_after(sim_grid, r, c, color, captured_any):
                return False

        # Ko checks
        if self._ko_rule is not None:
            new_turn = -self.turn
            new_hash = self._hash_from(sim_grid, new_turn)
            if self._ko_rule == 'simple':
                if rules.simple_ko_forbidden(self._position_history[-1], new_hash, self._position_history):
                    return False
            else:  # positional superko
                new_poshash = self._poshash_from(sim_grid)
                if rules.positional_superko_forbidden(new_poshash, self._psk_set):
                    return False
        return True

    def _apply_move_with_rules(self, move: int) -> bool:
        if move == self.pass_index:
            # handled in play()
            return True
        if not self._is_legal(move):
            return False
        N = self.size
        r, c = divmod(int(move), N)
        color = 1 if self.turn == 1 else -1
        opp = -color

        # Place stone
        self.grid[r, c] = color

        # Capture adjacent opponent groups with no liberties (using shared rules)
        captures = []
        visited_opp_groups = set()
        for nr, nc in rules.neighbors(self.size, r, c):
            if self.grid[nr, nc] == opp and (nr, nc) not in visited_opp_groups:
                group, libs = rules.collect_group_and_liberties(self.grid, nr, nc)
                visited_opp_groups.update(group)
                if len(libs) == 0:
                    for rr, cc in group:
                        captures.append((rr, cc, opp))
                        self.grid[rr, cc] = 0

        # Update capture counters
        if captures:
            if color == 1:
                self.captures_black += len(captures)
            else:
                self.captures_white += len(captures)

        self.history.append(int(move))
        # Flip turn and update position history/ko sets
        self.turn *= -1
        new_hash = self._hash_from(self.grid, self.turn)
        self._position_history.append(new_hash)
        new_poshash = self._poshash_from(self.grid)
        self._position_history_pos.append(new_poshash)
        if self._ko_rule == 'psk':
            self._psk_set.add(new_poshash)

        # Record for undo
        self._move_stack.append({
            'move': int(move),
            'placed_color': color,
            'captures': captures,
        })
        return True


class Board9x9(Board):
    def __init__(self):
        super().__init__(9)

