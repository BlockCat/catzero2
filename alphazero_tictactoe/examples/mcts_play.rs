use std::collections::HashMap;

use alphazero_tictactoe::{GameResult, Move, TicTacToe};
use mcts::{GameState, StateEvaluation, selection::StandardSelectionStrategy};
use rand::seq::IteratorRandom;

fn main() {
    println!("Random play example for Tic Tac Toe");
    let selection_strategy = mcts::selection::StandardSelectionStrategy::new(1.4);
    let state_evaluation = TicTacToeStateEvaluation;
    let mut mcts = mcts::MCTS::<
        TicTacToe,
        Move,
        mcts::DefaultAdjacencyTree<Move>,
        StandardSelectionStrategy,
        TicTacToeStateEvaluation,
    >::new(selection_strategy, state_evaluation, 0.9);

    let mut game = TicTacToe::new();
    game = game.make_move(Move::new(0, 0)); // Let X start in the center
    game = game.make_move(Move::new(1, 1)); // Let O play somewhere else

    while let GameResult::InProgress = game.check_winner() {
        // let best_move = mtcs
        //     .search_for_duration(&game, std::time::Duration::from_secs(4))
        //     .expect("Could not find move?");
        let best_move = mcts
            .search_for_iterations(&game, 600_000)
            .expect("Could not find move?");
        mcts.subtree_pruning(best_move.clone());
        game = game.make_move(best_move);

        println!(
            "Player {:?} plays move: {:?}",
            game.current_player().opponent(),
            best_move
        );

        println!("{}", game.display());
    }
    match game.check_winner() {
        GameResult::Win(player) => {
            println!("Player {:?} wins!", player);
        }
        GameResult::Draw => {
            println!("The game is a draw.");
        }
        GameResult::InProgress => unreachable!(),
    }
}

struct TicTacToeStateEvaluation;

impl StateEvaluation<TicTacToe> for TicTacToeStateEvaluation {
    async fn evaluation(&self, state: &TicTacToe, _: &[TicTacToe]) -> mcts::ModelEvaluation<Move> {
        let possible_actions = state.get_possible_actions();
        let action_count = possible_actions.len();

        let policy = possible_actions
            .into_iter()
            .map(|mv| (mv, 1.0 / action_count as f32))
            .collect::<HashMap<Move, f32>>();

        let value = match state.check_winner() {
            GameResult::Win(winner) => {
                debug_assert_eq!(winner, state.current_player().opponent());
                -1.0
            }
            GameResult::Draw => 0.0,
            GameResult::InProgress => {
                let simulated_result = random_play(state);
                match simulated_result {
                    GameResult::Win(winner) => {
                        if winner == state.current_player() {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    GameResult::Draw => 0.0,
                    GameResult::InProgress => unreachable!(),
                }
            }
        };

        mcts::ModelEvaluation::new(policy, value)
    }
}

fn random_play(game: &TicTacToe) -> GameResult {
    let mut rng = rand::rng();
    let mut current_game = game.clone();
    while let GameResult::InProgress = current_game.check_winner() {
        let available_moves: Vec<_> = current_game.get_possible_actions();
        if let Some(&mv) = available_moves.iter().choose(&mut rng) {
            current_game = current_game.make_move(mv);
        }
    }
    current_game.check_winner()
}
