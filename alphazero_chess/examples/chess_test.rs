use std::str::FromStr;

use alphazero_chess::ChessWrapper;

fn main() {
    let mut chess = chess::Game::new();

    println!(
        "Initial chess position:\n{}",
        ChessWrapper(chess.current_position()).pretty_print()
    );
    println!("status: {:?}", chess.current_position().status());

    for mv_str in ["f2f4", "e7e5", "g2g4", "d8h4"] {
        let mv = chess::ChessMove::from_str(mv_str).unwrap();
        chess.make_move(mv);

        println!(
            "Chess position:\n{}",
            ChessWrapper(chess.current_position()).pretty_print()
        );
        println!(
            "status: {:?}, pl: {:?}",
            chess.current_position().status(),
            chess.current_position().side_to_move()
        );
    }
}
