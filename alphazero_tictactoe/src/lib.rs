use mcts::GameState;

/// Represents a player in TicTacToe
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    X,
    O,
}

impl Player {
    /// Returns the opposite player
    pub fn opponent(self) -> Self {
        match self {
            Player::X => Player::O,
            Player::O => Player::X,
        }
    }
}

/// Represents a cell on the TicTacToe board
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    Empty,
    Occupied(Player),
}

/// Represents a move in TicTacToe (row, col)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub row: usize,
    pub col: usize,
}

impl Move {
    pub fn new(row: usize, col: usize) -> Self {
        Move { row, col }
    }

    /// Convert position (0-8) to Move
    pub fn from_position(pos: usize) -> Self {
        Move {
            row: pos / 3,
            col: pos % 3,
        }
    }

    /// Convert Move to position (0-8)
    pub fn to_position(&self) -> usize {
        self.row * 3 + self.col
    }
}

/// Represents the game outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Win(Player),
    Draw,
    InProgress,
}

/// TicTacToe game state
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TicTacToe {
    board: [[Cell; 3]; 3],
    current_player: Player,
    move_count: u8,
}

impl TicTacToe {
    /// Create a new TicTacToe game
    pub fn new() -> Self {
        TicTacToe {
            board: [[Cell::Empty; 3]; 3],
            current_player: Player::X,
            move_count: 0,
        }
    }

    /// Get the current player
    pub fn current_player(&self) -> Player {
        self.current_player
    }

    /// Get the cell at the given position
    pub fn get_cell(&self, row: usize, col: usize) -> Cell {
        self.board[row][col]
    }

    /// Check if a move is valid
    pub fn is_valid_move(&self, mv: Move) -> bool {
        mv.row < 3 && mv.col < 3 && self.board[mv.row][mv.col] == Cell::Empty
    }

    /// Make a move and return the new state
    pub fn make_move(&self, mv: Move) -> Self {
        assert!(self.is_valid_move(mv), "Invalid move");

        let mut new_state = self.clone();
        new_state.board[mv.row][mv.col] = Cell::Occupied(self.current_player);
        new_state.current_player = self.current_player.opponent();
        new_state.move_count += 1;
        new_state
    }

    /// Check for a winner and return the game result
    pub fn check_winner(&self) -> GameResult {
        // Check rows
        for row in 0..3 {
            if let Cell::Occupied(player) = self.board[row][0]
                && self.board[row][1] == Cell::Occupied(player)
                    && self.board[row][2] == Cell::Occupied(player)
                {
                    return GameResult::Win(player);
                }
        }

        // Check columns
        for col in 0..3 {
            if let Cell::Occupied(player) = self.board[0][col]
                && self.board[1][col] == Cell::Occupied(player)
                    && self.board[2][col] == Cell::Occupied(player)
                {
                    return GameResult::Win(player);
                }
        }

        // Check diagonals
        if let Cell::Occupied(player) = self.board[1][1] {
            // Main diagonal
            if self.board[0][0] == Cell::Occupied(player)
                && self.board[2][2] == Cell::Occupied(player)
            {
                return GameResult::Win(player);
            }
            // Anti-diagonal
            if self.board[0][2] == Cell::Occupied(player)
                && self.board[2][0] == Cell::Occupied(player)
            {
                return GameResult::Win(player);
            }
        }

        // Check for draw
        if self.move_count == 9 {
            return GameResult::Draw;
        }

        GameResult::InProgress
    }

    /// Display the board as a string
    pub fn display(&self) -> String {
        let mut result = String::new();
        for row in 0..3 {
            for col in 0..3 {
                let symbol = match self.board[row][col] {
                    Cell::Empty => ".",
                    Cell::Occupied(Player::X) => "X",
                    Cell::Occupied(Player::O) => "O",
                };
                result.push_str(symbol);
                if col < 2 {
                    result.push('|');
                }
            }
            if row < 2 {
                result.push_str("\n-+-+-\n");
            }
        }
        result
    }
}

impl Default for TicTacToe {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState for TicTacToe {
    type Action = Move;

    fn get_possible_actions(&self) -> Vec<Self::Action> {
        let mut actions = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                if self.board[row][col] == Cell::Empty {
                    actions.push(Move::new(row, col));
                }
            }
        }
        actions
    }

    fn take_action(&self, action: Self::Action) -> Self {
        self.make_move(action)
    }

    fn is_terminal(&self) -> Option<f64> {
        match self.check_winner() {
            GameResult::Win(player) => {
                // Return +1 if current player's opponent won (they made the winning move)
                // Return -1 if current player lost
                if player == self.current_player.opponent() {
                    Some(1.0)
                } else {
                    Some(-1.0)
                }
            }
            GameResult::Draw => Some(0.0),
            GameResult::InProgress => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = TicTacToe::new();
        assert_eq!(game.current_player(), Player::X);
        assert_eq!(game.check_winner(), GameResult::InProgress);
        assert_eq!(game.get_possible_actions().len(), 9);
    }

    #[test]
    fn test_make_move() {
        let game = TicTacToe::new();
        let game = game.make_move(Move::new(0, 0));
        assert_eq!(game.get_cell(0, 0), Cell::Occupied(Player::X));
        assert_eq!(game.current_player(), Player::O);
    }

    #[test]
    fn test_horizontal_win() {
        let game = TicTacToe::new();
        let game = game.make_move(Move::new(0, 0)); // X
        let game = game.make_move(Move::new(1, 0)); // O
        let game = game.make_move(Move::new(0, 1)); // X
        let game = game.make_move(Move::new(1, 1)); // O
        let game = game.make_move(Move::new(0, 2)); // X wins
        assert_eq!(game.check_winner(), GameResult::Win(Player::X));
    }

    #[test]
    fn test_vertical_win() {
        let game = TicTacToe::new();
        let game = game.make_move(Move::new(0, 0)); // X
        let game = game.make_move(Move::new(0, 1)); // O
        let game = game.make_move(Move::new(1, 0)); // X
        let game = game.make_move(Move::new(1, 1)); // O
        let game = game.make_move(Move::new(2, 0)); // X wins
        assert_eq!(game.check_winner(), GameResult::Win(Player::X));
    }

    #[test]
    fn test_diagonal_win() {
        let game = TicTacToe::new();
        let game = game.make_move(Move::new(0, 0)); // X
        let game = game.make_move(Move::new(0, 1)); // O
        let game = game.make_move(Move::new(1, 1)); // X
        let game = game.make_move(Move::new(0, 2)); // O
        let game = game.make_move(Move::new(2, 2)); // X wins
        assert_eq!(game.check_winner(), GameResult::Win(Player::X));
    }

    #[test]
    fn test_draw() {
        let game = TicTacToe::new();
        let game = game.make_move(Move::new(0, 0)); // X
        let game = game.make_move(Move::new(0, 1)); // O
        let game = game.make_move(Move::new(0, 2)); // X
        let game = game.make_move(Move::new(1, 0)); // O
        let game = game.make_move(Move::new(1, 1)); // X
        let game = game.make_move(Move::new(2, 2)); // O
        let game = game.make_move(Move::new(1, 2)); // X
        let game = game.make_move(Move::new(2, 0)); // O
        let game = game.make_move(Move::new(2, 1)); // X
        assert_eq!(game.check_winner(), GameResult::Draw);
    }

    #[test]
    fn test_is_valid_move() {
        let game = TicTacToe::new();
        assert!(game.is_valid_move(Move::new(0, 0)));
        let game = game.make_move(Move::new(0, 0));
        assert!(!game.is_valid_move(Move::new(0, 0)));
    }

    #[test]
    fn test_display() {
        let game = TicTacToe::new();
        let game = game.make_move(Move::new(0, 0)); // X
        let game = game.make_move(Move::new(1, 1)); // O
        let display = game.display();
        assert!(display.contains("X"));
        assert!(display.contains("O"));
    }

    #[test]
    fn test_game_state_trait() {
        let game = TicTacToe::new();

        // Test get_possible_actions
        let actions = game.get_possible_actions();
        assert_eq!(actions.len(), 9);

        // Test take_action
        let new_game = game.take_action(Move::new(0, 0));
        assert_eq!(new_game.get_cell(0, 0), Cell::Occupied(Player::X));

        // Test is_terminal on in-progress game
        assert_eq!(game.is_terminal(), None);

        // Test is_terminal on winning game
        let game = TicTacToe::new();
        let game = game.make_move(Move::new(0, 0)); // X
        let game = game.make_move(Move::new(1, 0)); // O
        let game = game.make_move(Move::new(0, 1)); // X
        let game = game.make_move(Move::new(1, 1)); // O
        let game = game.make_move(Move::new(0, 2)); // X wins
        assert_eq!(game.is_terminal(), Some(1.0));
    }
}
