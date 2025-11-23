use std::str::FromStr;

use alphazero_chess::ChessWrapper;
use chess::{Board, ChessMove, Color, Game};
use mcts::{GameState, StateEvaluation, selection::StandardSelectionStrategy};
use rand::seq::IteratorRandom;

fn main() {
    println!("Random play example for Tic Tac Toe");
    let selection_strategy = mcts::selection::StandardSelectionStrategy::new(1.4);
    let state_evaluation = ChessStateEvaluation;
    let mtcs = mcts::MCTS::<
        ChessWrapper,
        ChessMove,
        mcts::DefaultAdjacencyTree<ChessMove>,
        StandardSelectionStrategy,
        ChessStateEvaluation,
    >::new(selection_strategy, state_evaluation, 0.9);

    // let mut game = ChessWrapper::new();
    let mut game = Game::new();

    while let None = game.result() {
        println!("Player {:?} to move", game.side_to_move());

        let best_move = mtcs
            .search_for_duration(
                &ChessWrapper(game.current_position()),
                std::time::Duration::from_secs(4),
            )
            .expect("Could not find move?");
        // let best_move = mtcs
        //     .search_for_iterations(&game, 6000_000)
        //     .expect("Could not find move?");

        println!("Player {:?} plays move: {}", game.side_to_move(), best_move);

        game.make_move(best_move);

        println!("{}", ChessWrapper(game.current_position()).pretty_print());
    }

    match game.result() {
        Some(result) => match result {
            chess::GameResult::WhiteCheckmates => println!("White wins by checkmate!"),
            chess::GameResult::BlackCheckmates => println!("Black wins by checkmate!"),
            chess::GameResult::Stalemate => println!("Game ends in stalemate."),
            chess::GameResult::DrawAccepted | chess::GameResult::DrawDeclared => {
                println!("Game ends in a draw.")
            }
            chess::GameResult::WhiteResigns => unreachable!(),
            chess::GameResult::BlackResigns => unreachable!(),
        },
        _ => unreachable!(),
    }

    
}

struct ChessStateEvaluation;

impl StateEvaluation<ChessWrapper> for ChessStateEvaluation {
    async fn evaluation(&self, state: &ChessWrapper, previous_state: &[ChessWrapper]) -> mcts::ModelEvaluation {
        let possible_actions = state.get_possible_actions();
        let action_count = possible_actions.len();
        let policy = if action_count > 0 {
            let prob = 1.0 / action_count as f32;
            possible_actions.iter().map(|_| prob).collect::<Vec<f32>>()
        } else {
            Vec::new()
        };

        let value = match state.0.status() {
            chess::BoardStatus::Stalemate => 0.1,
            chess::BoardStatus::Checkmate => 1.0,
            chess::BoardStatus::Ongoing => random_play(state),
        };

        mcts::ModelEvaluation::new(policy, value)
    }
}

fn random_play(start_game: &ChessWrapper) -> f64 {
    let mut rng = rand::rng();
    let mut current_game = start_game.clone();

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

    return current_game.evaluate_position();
    // val
}
