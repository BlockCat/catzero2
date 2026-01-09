use alphazero_chess::ChessWrapper;
use chess::{ChessMove, Game};
use mcts::{GameState, Result, StateEvaluation, selection::StandardSelectionStrategy};

fn main() {
    println!("Random play example for Tic Tac Toe");
    let selection_strategy = mcts::selection::StandardSelectionStrategy::new(1.4);
    let state_evaluation = ChessStateEvaluation;
    let mut mtcs = mcts::MCTS::<
        ChessWrapper,
        mcts::DefaultAdjacencyTree<ChessMove>,
        StandardSelectionStrategy,
        ChessStateEvaluation,
    >::new(selection_strategy, state_evaluation, 0.9)
    .unwrap();

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

        println!("Known positions: {}", mtcs.positions_expanded());

        mtcs.subtree_pruning(best_move.clone()).unwrap();

        println!(
            "Known positions after pruning: {}",
            mtcs.positions_expanded()
        );
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
    async fn evaluation(
        &self,
        state: &ChessWrapper,
        _previous_state: &[ChessWrapper],
    ) -> Result<mcts::ModelEvaluation<ChessMove>> {
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

        Ok(mcts::ModelEvaluation::new(policy, value))
    }
}

fn random_play(start_game: &ChessWrapper) -> f32 {
    start_game.evaluate_position()
}
