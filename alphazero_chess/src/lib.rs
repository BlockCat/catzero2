use chess::{Board, MoveGen};
use mcts::GameState;

pub use chess;

#[derive(Clone)]
pub struct ChessWrapper(pub Board);

unsafe impl Send for ChessWrapper {}
unsafe impl Sync for ChessWrapper {}

impl Default for ChessWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl ChessWrapper {
    pub fn new() -> Self {
        ChessWrapper(Board::default())
    }

    pub fn pretty_print(&self) -> String {
        let mut board_str = String::new();
        for rank in (0..8).rev() {
            board_str.push_str(&format!("{} ", rank + 1));
            for file in 0..8 {
                let square = chess::Square::make_square(
                    chess::Rank::from_index(rank),
                    chess::File::from_index(file),
                );
                let piece = self.0.piece_on(square);
                let symbol = match piece {
                    Some(p) => {
                        let color = self.0.color_on(square).unwrap();
                        match (p, color) {
                            (chess::Piece::Pawn, chess::Color::White) => '♙',
                            (chess::Piece::Knight, chess::Color::White) => '♘',
                            (chess::Piece::Bishop, chess::Color::White) => '♗',
                            (chess::Piece::Rook, chess::Color::White) => '♖',
                            (chess::Piece::Queen, chess::Color::White) => '♕',
                            (chess::Piece::King, chess::Color::White) => '♔',
                            (chess::Piece::Pawn, chess::Color::Black) => '♟',
                            (chess::Piece::Knight, chess::Color::Black) => '♞',
                            (chess::Piece::Bishop, chess::Color::Black) => '♝',
                            (chess::Piece::Rook, chess::Color::Black) => '♜',
                            (chess::Piece::Queen, chess::Color::Black) => '♛',
                            (chess::Piece::King, chess::Color::Black) => '♚',
                        }
                    }
                    None => '.',
                };
                board_str.push(symbol);
                board_str.push(' ');
            }
            board_str.push('\n');
        }
        board_str.push_str("  a b c d e f g h\n");
        board_str
    }

    pub fn evaluate_position(&self) -> f32 {
        // Simple evaluation: material count

        if self.0.status() == chess::BoardStatus::Checkmate {
            return 1.0;
        } else if self.0.status() == chess::BoardStatus::Stalemate {
            return 0.0;
        }

        let mut score = 0.0;
        for square in chess::ALL_SQUARES {
            if let Some(piece) = self.0.piece_on(square) {
                let color = self.0.color_on(square).unwrap();
                let piece_value = match piece {
                    chess::Piece::Pawn => 1.0,   //8
                    chess::Piece::Knight => 3.0, // 6
                    chess::Piece::Bishop => 3.0, // 6
                    chess::Piece::Rook => 5.0,   // 10
                    chess::Piece::Queen => 9.0,  // 9
                    chess::Piece::King => 0.0,
                };
                // max score: 8+6+6+10+9=39

                if color == self.0.side_to_move() {
                    score -= piece_value;
                } else {
                    score += piece_value;
                }
            }
        }

        score / 50.0
    }
}

impl GameState for ChessWrapper {
    type Action = chess::ChessMove;

    fn current_player_id(&self) -> usize {
        if self.0.side_to_move() == chess::Color::White {
            0
        } else {
            1
        }
    }

    fn get_possible_actions(&self) -> Vec<Self::Action> {
        MoveGen::new_legal(&self.0).collect()
    }

    fn take_action(&self, action: Self::Action) -> Self {
        let new_board = self.0.clone().make_move_new(action);
        ChessWrapper(new_board)
    }

    fn is_terminal(&self) -> Option<f32> {
        if self.0.status() == chess::BoardStatus::Checkmate {
            Some(1000.0)
        } else if self.0.status() == chess::BoardStatus::Stalemate {
            Some(0.0)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::ChessMove;
    use mcts::{StateEvaluation, selection::StandardSelectionStrategy};
    use std::str::FromStr as _;

    #[test]
    fn test_initial_possible_moves() {
        let game = ChessWrapper::new();
        let possible_moves = game.get_possible_actions();
        assert_eq!(possible_moves.len(), 20); // 16 pawn moves + 4 knight moves

        assert_eq!(game.0, Board::default());

        println!("{}", game.pretty_print());
    }

    #[test]
    fn test_find_mate_in_one() {
        let selection_strategy = StandardSelectionStrategy::new(1.4);
        let state_evaluation = ChessStateEvaluation;
        let mut mtcs = mcts::MCTS::<
            ChessWrapper,
            ChessMove,
            mcts::DefaultAdjacencyTree<ChessMove>,
            StandardSelectionStrategy,
            ChessStateEvaluation,
        >::new(selection_strategy, state_evaluation, 0.9)
        .unwrap();
        let mut game = ChessWrapper::new();
        game.0 = Board::from_str("r5r1/p2nR3/2k5/P1pn3p/7P/1RPP1PB1/5P2/5K2 w - - 1 32").unwrap();

        let best_move = mtcs
            .search_for_iterations(&game, 10_000)
            .expect("Could not find move?");
        assert_eq!(best_move.to_string(), "e7e6");
    }

    #[test]
    fn test_find_mate_in_two() {
        let selection_strategy = StandardSelectionStrategy::new(2.4);
        let state_evaluation = ChessStateEvaluation;
        let mut mtcs = mcts::MCTS::<
            ChessWrapper,
            ChessMove,
            mcts::DefaultAdjacencyTree<ChessMove>,
            StandardSelectionStrategy,
            ChessStateEvaluation,
        >::new(selection_strategy, state_evaluation, 0.9)
        .unwrap();
        let mut game = ChessWrapper::new();
        game.0 =
            Board::from_str("rn3rk1/pp3pp1/2pbb3/4N3/2PPN3/8/PPQ2Pq1/R3K2R w KQ - 0 15").unwrap();

        let best_move = mtcs
            .search_for_iterations(&game, 10_000)
            .expect("Could not find move?");
        println!("Known positions: {}", mtcs.positions_expanded());
        mtcs.subtree_pruning(best_move.clone()).unwrap();
        println!(
            "Known positions after pruning: {}",
            mtcs.positions_expanded()
        );
        assert_eq!(best_move.to_string(), "e4f6");

        game = game.take_action(best_move);
        let best_move = mtcs
            .search_for_iterations(&game, 10_000)
            .expect("Could not find move?");
        println!("Known positions: {}", mtcs.positions_expanded());
        mtcs.subtree_pruning(best_move.clone()).unwrap();
        println!(
            "Known positions after pruning: {}",
            mtcs.positions_expanded()
        );
        assert_eq!(best_move.to_string(), "g7f6");

        game = game.take_action(best_move);
        let best_move = mtcs
            .search_for_iterations(&game, 10_000)
            .expect("Could not find move?");
        assert_eq!(best_move.to_string(), "c2h7");
    }

    struct ChessStateEvaluation;

    impl StateEvaluation<ChessWrapper> for ChessStateEvaluation {
        async fn evaluation(
            &self,
            state: &ChessWrapper,
            _previous_state: &[ChessWrapper],
        ) -> mcts::ModelEvaluation<ChessMove> {
            let possible_actions = state.get_possible_actions();
            let action_count = possible_actions.len();

            let policy = possible_actions
                .into_iter()
                .map(|action| (action, 1.0 / action_count as f32))
                .collect::<std::collections::HashMap<_, _>>();

            let value = match state.0.status() {
                chess::BoardStatus::Stalemate => 0.1,
                chess::BoardStatus::Checkmate => 1.0,
                chess::BoardStatus::Ongoing => random_play(state),
            };

            mcts::ModelEvaluation::new(policy, value)
        }
    }

    fn random_play(start_game: &ChessWrapper) -> f32 {
        // let mut rng = rand::rng();s
        // let current_game = start_game.clone();

        // let mut counter = 0;
        // while let None = current_game.is_terminal() {
        //     let available_moves: Vec<_> = current_game.get_possible_actions();
        //     if let Some(&mv) = available_moves.iter().choose(&mut rng) {
        //         current_game = current_game.take_action(mv);
        //     } else {
        //         break;
        //     }
        //     counter += 1;
        //     if counter > 100 {
        //         return if current_game.0.side_to_move() == start_game.0.side_to_move() {
        //             current_game.evaluate_position()
        //         } else {
        //             -current_game.evaluate_position()
        //         };
        //     }
        // }

        // let val = if current_game.0.side_to_move() == start_game.0.side_to_move() {
        //     current_game.evaluate_position()
        // } else {
        //     -current_game.evaluate_position()
        // };

        return start_game.evaluate_position();
        // val
    }
}
