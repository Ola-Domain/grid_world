GRID_SIZE = 5
CELL_SIZE = 100
PADDING = 80
WIDTH = GRID_SIZE * CELL_SIZE + PADDING * 2
HEIGHT = GRID_SIZE * CELL_SIZE + PADDING * 2 + 60  # Space for reward

START_POS = (2, 1)
TERMINAL_STATE = (5, 5)
SPECIAL_JUMP = {(2, 4): (4, 4)}
OBSTACLES = [(3, 3), (4,3), (3, 5), (3, 4)]

ACTIONS = {
    1: (-1, 0),  # North
    2: (1, 0),   # South
    3: (0, 1),   # East
    4: (0, -1),  # West
}
