use alphazero_tictactoe::{GameResult, TicTacToe};
use mcts::GameState;
use rand::seq::IteratorRandom;

fn main() {
    println!("Random play example for Tic Tac Toe");
    let mut game = TicTacToe::new();
    let mut rng = rand::rng();
    while let GameResult::InProgress = game.check_winner() {
        let available_moves: Vec<_> = game.get_possible_actions();
        if let Some(&mv) = available_moves.iter().choose(&mut rng) {
            game = game.make_move(mv);
            println!(
                "Player {:?} plays move: {:?}",
                game.current_player().opponent(),
                mv
            );
            println!("{}", game.display());
        }
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
