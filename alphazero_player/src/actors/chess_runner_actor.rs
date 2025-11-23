use actix::{dev::ContextFutureSpawner, Actor, Context, Recipient, WrapFuture};
use alphazero_chess::{
    chess::{BitBoard, ChessMove, Color, Piece},
    ChessWrapper,
};
use candle_core::{Device, Tensor};
use mcts::{
    AlphaZeroSelectionStrategy, DefaultAdjacencyTree, GameState, ModelEvaluation, StateEvaluation,
    MCTS,
};

use crate::actors::batch_actor::InferenceRequest;

pub struct ChessRunnerActor {
    mcts: MCTS<
        ChessWrapper,
        ChessMove,
        DefaultAdjacencyTree<ChessMove>,
        AlphaZeroSelectionStrategy,
        ActorAlphaEvaluator,
    >,
    iters: usize,
}

impl ChessRunnerActor {
    pub fn new(
        c1: f32,
        c2: f32,
        discount_factor: f64,
        iters: usize,
        batcher: Recipient<InferenceRequest>,
        device: Device,
    ) -> Self {
        let selection_strategy = AlphaZeroSelectionStrategy::new(c1, c2);
        let state_evaluation = ActorAlphaEvaluator {
            historic_moves: 8,
            batcher,
            device,
        };
        let mcts = MCTS::new(selection_strategy, state_evaluation, discount_factor);

        ChessRunnerActor { mcts, iters }
    }
}

impl Actor for ChessRunnerActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        async move {
            let mut game = ChessWrapper::new();
            let mut mcts = self.mcts.clone();

            let mut positions: Vec<(ChessWrapper, Vec<(ChessMove, f32)>)> = Vec::with_capacity(40);

            loop {
                let best_move = mcts
                    .search_for_iterations_async(&game, self.iters)
                    .await
                    .expect("MCTS did not return a move");

                positions.push((game.clone(), mcts.get_action_probabilities()));

                game = game.take_action(best_move.clone());

                mcts.subtree_pruning(best_move.clone());

                println!("Played move: {}", best_move);
                println!("{}", game.0);
            }
        }
        .into_actor(self)
        .spawn(ctx);
    }
}

#[derive(Clone)]
struct ActorAlphaEvaluator {
    historic_moves: usize,
    batcher: Recipient<InferenceRequest>,
    device: Device,
}

impl StateEvaluation<ChessWrapper> for ActorAlphaEvaluator {
    async fn evaluation(
        &self,
        state: &ChessWrapper,
        previous_state: &[ChessWrapper],
    ) -> ModelEvaluation {
        let standard_planes = 6; // ignoring no progress plane for now
        let board_plane = 12; // 12 piece planes, we ignore repetition planes for now
        let total_planes = board_plane * self.historic_moves + standard_planes;

        let tensor = {
            let mut slice = vec![0.0f32; 8 * 8 * (total_planes + standard_planes)];
            add_state(&mut slice[0..8 * 8 * board_plane], state);

            previous_state
                .iter()
                .rev()
                .take(self.historic_moves - 1)
                .enumerate()
                .for_each(|(i, prev_state)| {
                    let start = (i + 1) * board_plane * 8 * 8;
                    let end = start + board_plane * 8 * 8;
                    add_state(&mut slice[start..end], prev_state);
                });

            let standard_plane_start = self.historic_moves * board_plane * 8 * 8usize;
            let plane_size = 8 * 8usize;

            let colour = match state.0.side_to_move() {
                Color::White => 1.0f32,
                Color::Black => 0.0f32,
            };
            let move_count = previous_state.len() as f32;
            let white_castle = state.0.castle_rights(Color::White);
            let black_castle = state.0.castle_rights(Color::Black);

            slice[standard_plane_start..standard_plane_start + plane_size]
                .copy_from_slice(&vec![colour; plane_size]);
            slice[standard_plane_start + plane_size..standard_plane_start + 2 * plane_size]
                .copy_from_slice(&vec![move_count; plane_size]);
            slice[standard_plane_start + 2 * plane_size..standard_plane_start + 3 * plane_size]
                .copy_from_slice(&vec![white_castle.has_kingside() as i32 as f32; plane_size]);
            slice[standard_plane_start + 3 * plane_size..standard_plane_start + 4 * plane_size]
                .copy_from_slice(&vec![
                    white_castle.has_queenside() as i32 as f32;
                    plane_size
                ]);
            slice[standard_plane_start + 4 * plane_size..standard_plane_start + 5 * plane_size]
                .copy_from_slice(&vec![black_castle.has_kingside() as i32 as f32; plane_size]);
            slice[standard_plane_start + 5 * plane_size..standard_plane_start + 6 * plane_size]
                .copy_from_slice(&vec![
                    black_castle.has_queenside() as i32 as f32;
                    plane_size
                ]);

            Tensor::from_slice::<(usize, usize, usize), f32>(
                &slice,
                (total_planes, 8, 8),
                &self.device,
            )
            .expect("Could not create tensor")
        };

        let player_id = state.0.side_to_move().to_index() as u32;
        let response = self
            .batcher
            .send(InferenceRequest {
                player_id,
                input: tensor,
            })
            .await
            .expect("Failed to send InferenceRequest");

        ModelEvaluation::new(vec![], response.value as f64)
    }
}

// We ignore repetition planes for now
fn add_state(slice: &mut [f32], state: &ChessWrapper) {
    let pawns = state.0.pieces(Piece::Pawn);
    let knights = state.0.pieces(Piece::Knight);
    let bishops = state.0.pieces(Piece::Bishop);
    let rooks = state.0.pieces(Piece::Rook);
    let queens = state.0.pieces(Piece::Queen);
    let kings = state.0.pieces(Piece::King);

    let white = state.0.color_combined(Color::White);
    let black = state.0.color_combined(Color::Black);

    let white_pawns = pawns & white;
    let white_knights = knights & white;
    let white_bishops = bishops & white;
    let white_rooks = rooks & white;
    let white_queens = queens & white;
    let white_kings = kings & white;
    let black_pawns = pawns & black;
    let black_knights = knights & black;
    let black_bishops = bishops & black;
    let black_rooks = rooks & black;
    let black_queens = queens & black;
    let black_kings = kings & black;

    bitboard_to_plane(&mut slice[0..8 * 8], white_pawns);
    bitboard_to_plane(&mut slice[8 * 8..2 * 8 * 8], white_knights);
    bitboard_to_plane(&mut slice[2 * 8 * 8..3 * 8 * 8], white_bishops);
    bitboard_to_plane(&mut slice[3 * 8 * 8..4 * 8 * 8], white_rooks);
    bitboard_to_plane(&mut slice[4 * 8 * 8..5 * 8 * 8], white_queens);
    bitboard_to_plane(&mut slice[5 * 8 * 8..6 * 8 * 8], white_kings);
    bitboard_to_plane(&mut slice[6 * 8 * 8..7 * 8 * 8], black_pawns);
    bitboard_to_plane(&mut slice[7 * 8 * 8..8 * 8 * 8], black_knights);
    bitboard_to_plane(&mut slice[8 * 8 * 8..9 * 8 * 8], black_bishops);
    bitboard_to_plane(&mut slice[9 * 8 * 8..10 * 8 * 8], black_rooks);
    bitboard_to_plane(&mut slice[10 * 8 * 8..11 * 8 * 8], black_queens);
    bitboard_to_plane(&mut slice[11 * 8 * 8..12 * 8 * 8], black_kings);
}

fn bitboard_to_plane(plane: &mut [f32], bitboard: BitBoard) {
    let u = bitboard.0;

    (0..64).for_each(|i| {
        if (u >> i) & 1 == 1 {
            plane[i as usize] = 1.0;
        }
    });
}
