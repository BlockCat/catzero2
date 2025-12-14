use crate::{
    inference::{InferenceRequest, InferenceService},
    runner::{GamePlayed, RunnerError, SingleRunner},
};
use alphazero_chess::{
    chess::{self, BitBoard, ChessMove, Color, Piece},
    ChessWrapper,
};
use candle_core::{Device, Shape, Tensor};
use mcts::{DefaultAdjacencyTree, GameState, ModelEvaluation, StateEvaluation, MCTS};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc::{self};

const STANDARD_PLANES: usize = 6; // ignoring no progress plane for now
const BOARD_STATE_PLANES: usize = 12; // 12 piece planes, we ignore repetition planes for now

#[derive(Debug, Clone)]
pub struct ChessConfig {
    pub num_iterations: usize,
    pub discount_factor: f64,
    pub c1: f32,
    pub c2: f32,
}

#[derive(Debug, Clone)]
pub struct ChessRunner {
    config: ChessConfig,
    channel: mpsc::Sender<GamePlayed<ChessWrapper>>,
    batcher: Arc<InferenceService>,
    device: Device,
}

impl ChessRunner {
    pub fn new(
        config: ChessConfig,
        channel: mpsc::Sender<GamePlayed<ChessWrapper>>,
        batcher: Arc<InferenceService>,
        device: Device,
    ) -> Self {
        ChessRunner {
            config,
            channel,
            batcher,
            device,
        }
    }
}

impl SingleRunner for ChessRunner {
    type GameState = ChessWrapper;

    async fn play_game(
        &self,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) -> Result<GamePlayed<Self::GameState>, RunnerError> {
        let selection_strategy =
            mcts::AlphaZeroSelectionStrategy::new(self.config.c1, self.config.c2);
        let state_evaluation = ChessActorAlphaEvaluator {
            historic_moves: 8,
            batcher: self.batcher.clone(),
            device: self.device.clone(),
        };

        let mut game = ChessWrapper::new();

        let mut states = Vec::new();
        let mut policies = Vec::new();
        let mut taken_actions = Vec::new();

        let mut mcts: MCTS<
            ChessWrapper,
            ChessMove,
            DefaultAdjacencyTree<ChessMove>,
            mcts::AlphaZeroSelectionStrategy,
            ChessActorAlphaEvaluator,
        > = MCTS::new(
            selection_strategy,
            state_evaluation,
            self.config.discount_factor,
        );

        while game.is_terminal().is_none() {
            if cancellation_token.is_cancelled() {
                return Err(RunnerError::Cancellation);
            }

            let best_move = mcts
                .search_for_iterations_async(&game, self.config.num_iterations)
                .await
                .expect("MCTS did not return a move");

            states.push(game.clone());
            policies.push(mcts.get_action_probabilities());
            taken_actions.push(best_move);

            game = game.take_action(best_move);

            mcts.subtree_pruning(best_move);
        }

        states.push(game.clone());

        let winner = match game.0.status() {
            chess::BoardStatus::Checkmate => Some(game.0.side_to_move().to_index() as i32),
            chess::BoardStatus::Stalemate => None,
            _ => unreachable!(),
        };

        Ok(GamePlayed {
            states,
            policies,
            taken_actions,
            value: 0.0,
            winner,
        })
    }

    async fn send_game_played(
        &self,
        game_played: GamePlayed<Self::GameState>,
    ) -> Result<(), RunnerError> {
        self.channel
            .send(game_played)
            .await
            .map_err(|e| RunnerError::GameError(anyhow::anyhow!(e)))?;
        Ok(())
    }

    fn tensor_input_shape(historic_states: usize) -> candle_core::Shape {
        Shape::from_dims(&[BOARD_STATE_PLANES * historic_states + STANDARD_PLANES, 8, 8])
    }
}

struct ChessActorAlphaEvaluator {
    pub historic_moves: usize,
    pub batcher: Arc<InferenceService>,
    pub device: Device,
}

impl StateEvaluation<ChessWrapper> for ChessActorAlphaEvaluator {
    async fn evaluation(
        &self,
        state: &ChessWrapper,
        previous_state: &[ChessWrapper],
    ) -> ModelEvaluation<ChessMove> {
        let state_tensor = {
            let shape = ChessRunner::tensor_input_shape(self.historic_moves);

            let mut slice = vec![0.0f32; shape.elem_count()];
            add_state(&mut slice[0..8 * 8 * BOARD_STATE_PLANES], state);

            previous_state
                .iter()
                .rev()
                .take(self.historic_moves - 1)
                .enumerate()
                .for_each(|(i, prev_state)| {
                    let start = (i + 1) * BOARD_STATE_PLANES * 8 * 8;
                    let end = start + BOARD_STATE_PLANES * 8 * 8;
                    add_state(&mut slice[start..end], prev_state);
                });

            let standard_plane_start = self.historic_moves * BOARD_STATE_PLANES * 8 * 8usize;
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

            Tensor::from_slice(&slice, shape, &self.device).expect("Could not create tensor")
        };

        let player_id = state.0.side_to_move().to_index() as u32;

        let response = self
            .batcher
            .request(InferenceRequest {
                player_id,
                state_tensor,
            })
            .await;

        // Convert flat policy tensor to sparse policy vector for legal moves
        let possible_actions = state.get_possible_actions();
        let policy_vec =
            convert_policy_tensor_to_action_probs(&response.output_tensor, &possible_actions)
                .expect("Failed to convert policy tensor");

        ModelEvaluation::new(policy_vec, response.value as f64)
    }
}

/// Convert flat policy tensor output to sparse policy vector for legal moves
/// The network outputs a flat tensor of size 4672 (64 squares × 73 move types)
/// We extract only the probabilities for legal moves in the current position
fn convert_policy_tensor_to_action_probs(
    _policy_tensor: &Tensor,
    _legal_moves: &[ChessMove],
) -> Result<HashMap<ChessMove, f32>, candle_core::Error> {
    todo!()
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
