use crate::{
    inference::InferenceService,
    runner::{alpha_evaluator::AlphaEvaluator, AlphaConfigurable, AlphaRunnable},
};
use alphazero_chess::{
    chess::{BitBoard, ChessMove, Color, Piece},
    ChessWrapper,
};
use alphazero_nn::{AlphaGame, PolicyOutputType};
use candle_core::{Device, Shape, Tensor};
use core::panic;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::RwLock;

/// Number of standard planes (current player, move count, castling rights (white/black) x (kingside/queenside))
const STANDARD_PLANES: usize = 6; // ignoring no progress plane for now
/// Number of planes per board state in history
const BOARD_STATE_PLANES: usize = 12; // 12 piece planes, we ignore repetition planes for now

pub struct Chess;

impl AlphaRunnable for Chess {
    type Config = ChessConfig;

    type StateEvaluation = AlphaEvaluator<Chess>;

    fn create_evaluator(config: &Self::Config) -> Self::StateEvaluation {
        AlphaEvaluator::new(config.inference_service.clone(), config.device.clone())
    }

    fn create_selector(config: &Self::Config) -> mcts::AlphaZeroSelectionStrategy {
        mcts::AlphaZeroSelectionStrategy::new(config.c1, config.c2)
    }
}

impl AlphaGame for Chess {
    type MoveType = alphazero_chess::chess::ChessMove;

    type GameState = alphazero_chess::ChessWrapper;

    fn tensor_input_shape() -> Shape {
        Shape::from_dims(&[
            BOARD_STATE_PLANES * Self::history_length() + STANDARD_PLANES,
            8,
            8,
        ])
    }

    /// Policy output type: convolutional with 73 planes (one per move type) for chess.
    /// Each plane is 8x8, with the from-square as spatial location.
    /// 8 planes x 7 distances for queen-like moves (N, NE, E, SE, S, SW, W, NW)
    /// 1 plane x 8 knight moves
    /// 3 planes x 3 underpromotion moves (Knight, Bishop, Rook) x (Forward, Left, Right), defaulting to Queen.
    /// Total: 73 planes.
    fn policy_output_type() -> PolicyOutputType {
        PolicyOutputType::Conv(73)
    }

    fn history_length() -> usize {
        20
    }

    fn encode_game_state(states: &[Self::GameState], _device: &Device) -> candle_core::Tensor {
        let shape = Self::tensor_input_shape();
        let mut slice = vec![0.0f32; shape.elem_count()];

        // States are ordered oldest..current, with the last element being the current state.
        // We encode the current state first, then history backwards.
        let history_to_encode = Self::history_length().min(states.len());
        for (i, state) in states.iter().rev().take(history_to_encode).enumerate() {
            let start = i * BOARD_STATE_PLANES * 8 * 8;
            let end = start + BOARD_STATE_PLANES * 8 * 8;
            add_state(&mut slice[start..end], state);
        }

        // Standard planes come after the full history-length block, regardless of how many
        // history states were available (missing history stays zero).
        let current_state = states
            .last()
            .expect("encode_game_state called with empty states");
        let standard_plane_start = Self::history_length() * BOARD_STATE_PLANES * 8 * 8;
        let plane_size = 8 * 8;

        let colour = match current_state.0.side_to_move() {
            Color::White => 1.0f32,
            Color::Black => 0.0f32,
        };
        let move_count = (states.len().saturating_sub(1)) as f32;
        let white_castle = current_state.0.castle_rights(Color::White);
        let black_castle = current_state.0.castle_rights(Color::Black);

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

        Tensor::from_slice(&slice, shape, _device).expect("Could not create tensor")
    }

    fn decode_policy_tensor(
        policy_tensor: &candle_core::Tensor,
        legal_moves: &[Self::MoveType],
    ) -> Result<HashMap<Self::MoveType, f32>, candle_core::Error> {
        convert_policy_tensor_to_action_probs(policy_tensor, legal_moves)
    }
}

#[derive(Debug, Clone)]
pub struct ChessConfig {
    pub num_iterations: usize,
    pub device: Device,
    pub inference_service: Arc<RwLock<InferenceService>>,
    pub discount_factor: f32,
    pub c1: f32,
    pub c2: f32,
}

impl AlphaConfigurable for ChessConfig {
    fn discount_factor(&self) -> f32 {
        self.discount_factor
    }

    fn num_iterations(&self) -> usize {
        self.num_iterations
    }

    fn c1(&self) -> f32 {
        self.c1
    }
    fn c2(&self) -> f32 {
        self.c2
    }
}

/// Convert flat policy tensor output to sparse policy vector for legal moves
/// The network outputs a flat tensor of size 4672 (64 squares × 73 move types)
/// We extract only the probabilities for legal moves in the current position
fn convert_policy_tensor_to_action_probs(
    policy_tensor: &Tensor,
    legal_moves: &[ChessMove],
) -> Result<HashMap<ChessMove, f32>, candle_core::Error> {
    let numel: usize = policy_tensor.shape().elem_count();
    let flat = policy_tensor.reshape(Shape::from_dims(&[numel]))?;
    let logits = flat.to_vec1::<f32>()?;

    let mut legal_logits = Vec::with_capacity(legal_moves.len());
    let mut legal_indices = Vec::with_capacity(legal_moves.len());

    for mv in legal_moves {
        if let Some(idx) = move_to_policy_index(mv) {
            if idx < logits.len() {
                legal_indices.push(mv);
                legal_logits.push(logits[idx]);
            } else {
                panic!(
                    "Policy index {} for move {:?} out of bounds (logits length {})",
                    idx,
                    mv,
                    logits.len()
                );
            }
        }
    }

    if legal_indices.is_empty() {
        return Ok(HashMap::new());
    }

    // Stable softmax over legal moves only.
    let max_logit = legal_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let mut exps = Vec::with_capacity(legal_logits.len());
    let mut sum = 0.0f32;
    for &l in &legal_logits {
        let e = (l - max_logit).exp();
        sum += e;
        exps.push(e);
    }

    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

    let mut out = HashMap::with_capacity(legal_indices.len());
    for (mv, e) in legal_indices.into_iter().zip(exps.into_iter()) {
        out.insert(*mv, e * inv_sum);
    }

    Ok(out)
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

/// Layer 0-6: N moves, distance 1-7
/// Layer 7-13: NE moves, distance 1-7
/// Layer 14-20: E moves, distance 1-7
/// Layer 21-27: SE moves, distance 1-7
/// Layer 28-34: S moves, distance 1-7
/// Layer 35-41: SW moves, distance 1-7
/// Layer 42-48: W moves, distance 1-7
/// Layer 49-55: NW moves, distance 1-7
/// Layer 56-63: Knight moves (8 directions)
/// Layer 64-72: Underpromotions (3x3 grid: Knight, Bishop, Rook) x (Left, Forward, Right), defaulting to Queen.
fn move_to_policy_index(mv: &ChessMove) -> Option<usize> {
    const LAYER_SIZE: usize = 8 * 8;

    let source_rank = mv.get_source().get_rank();
    let source_file = mv.get_source().get_file();
    let target_rank = mv.get_dest().get_rank();
    let target_file = mv.get_dest().get_file();
    let source_index = mv.get_source().to_index();

    if source_rank == target_rank && source_file == target_file {
        unreachable!(
            "Invalid move {:?}: source and target squares are the same",
            mv
        );
    }

    if let Some(promo) = mv.get_promotion() {
        let promotion_left_right_offset = if source_file == target_file {
            1 // Forward
        } else if target_file < source_file {
            0 // Left
        } else {
            2 // Right
        };
        // queen promotions are defaulted to normal move planes
        if promo != Piece::Queen {
            let promo_offset = match promo {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                Piece::Rook => 2,
                _ => unreachable!(
                    "Should not reach here: only Knight, Bishop, Rook allowed but was {:?}",
                    promo
                ),
            };
            let layer = 64 + promo_offset * 3 + promotion_left_right_offset;
            return Some(layer * LAYER_SIZE + source_index);
        }
    }
    let rank_diff = target_rank.to_index() as isize - source_rank.to_index() as isize;
    let file_diff = target_file.to_index() as isize - source_file.to_index() as isize;

    // Is knight move?
    if (rank_diff.abs() == 2 && file_diff.abs() == 1)
        || (rank_diff.abs() == 1 && file_diff.abs() == 2)
    {
        let offset = match (rank_diff, file_diff) {
            // NNE
            (2, 1) => 0,
            // NEE
            (1, 2) => 1,
            // SEE
            (-1, 2) => 2,
            // SSE
            (-2, 1) => 3,
            // SSW
            (-2, -1) => 4,
            // SWW
            (-1, -2) => 5,
            // NWW
            (1, -2) => 6,
            // NNW
            (2, -1) => 7,
            _ => unreachable!(),
        };
        let layer = 56 + offset;
        return Some(layer * LAYER_SIZE + source_index);
    }

    // Is queen-like move?
    if rank_diff == file_diff || rank_diff == 0 || file_diff == 0 {
        let distance = rank_diff.abs().max(file_diff.abs()) as usize;
        if distance == 0 || distance > 7 {
            unreachable!("Invalid queen-like move distance: {}", distance);
        }
        let direction = match (rank_diff.signum(), file_diff.signum()) {
            (1, 0) => 0,   // N
            (1, 1) => 1,   // NE
            (0, 1) => 2,   // E
            (-1, 1) => 3,  // SE
            (-1, 0) => 4,  // S
            (-1, -1) => 5, // SW
            (0, -1) => 6,  // W
            (1, -1) => 7,  // NW
            _ => unreachable!(),
        };
        let layer = direction * 7 + (distance - 1);
        return Some(layer * LAYER_SIZE + source_index);
    }

    unreachable!("Move {:?} could not be mapped to policy index", mv);
}
#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use alphazero_chess::chess::{File, Rank, Square};
    use candle_core::DType;

    fn approx_eq(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "{a} != {b}");
    }

    #[test]
    fn test_move_to_policy_index_queen_like_north() {
        let from = Square::from_str("a1").unwrap();
        for d in 1..=7 {
            let to_rank = Rank::from_index(d);
            let to = Square::make_square(to_rank, File::A);
            let mv = ChessMove::new(from, to, None);

            let idx = move_to_policy_index(&mv).expect("index");
            // North direction (dir=0), distance=d => plane=d-1.
            let expected_plane = d - 1;
            let expected = expected_plane * 64 + 0; // from a1=0
            assert_eq!(idx, expected);
        }
    }

    #[test]
    fn test_move_to_policy_index_queen_like_northeast() {
        let from = Square::from_str("a1").unwrap();
        for d in 1..=7 {
            let to_rank = Rank::from_index(d);
            let to_file = File::from_index(d);
            let to = Square::make_square(to_rank, to_file);
            let mv = ChessMove::new(from, to, None);

            let idx = move_to_policy_index(&mv).expect("index");
            // Northeast direction (dir=1), distance=d => plane=7 + (d-1)=d+6.
            let expected_plane = (d - 1) + 7;
            let expected = expected_plane * 64 + 0; // from a1=0
            assert_eq!(idx, expected);
        }
    }

    #[test]
    fn test_move_to_policy_index_knight() {
        let from = Square::make_square(Rank::First, File::B);
        let to = Square::make_square(Rank::Third, File::C);
        let mv = ChessMove::new(from, to, None);

        let idx = move_to_policy_index(&mv).expect("index");
        // (dx,dy)=(1,2) => first knight plane.
        let expected_plane = 56usize;
        let expected = expected_plane * 64 + from.to_index();
        assert_eq!(idx, expected);
    }

    #[test]
    fn test_move_to_policy_index_underpromotion() {
        let from = Square::from_str("a7").unwrap();
        let to = Square::from_str("a8").unwrap();
        let mv = ChessMove::new(from, to, Some(Piece::Rook));

        let idx = move_to_policy_index(&mv).expect("index");
        // Underpromotion: rook offset=2, straight dir=1 => plane=64 + 2*3 + 1 = 71
        let expected_plane = 71usize;
        let expected = expected_plane * 64 + from.to_index();
        assert_eq!(idx, expected);
    }

    #[test]
    fn test_decode_policy_tensor_softmax_over_legal_moves_only() {
        let device = Device::Cpu;
        let shape = Shape::from_dims(&[73, 8, 8]);
        let mut logits = vec![0.0f32; 73 * 8 * 8];

        let from1 = Square::make_square(Rank::Second, File::E);
        let to1 = Square::make_square(Rank::Fourth, File::E);
        let mv1 = ChessMove::new(from1, to1, None);
        let idx1 = move_to_policy_index(&mv1).unwrap();
        logits[idx1] = 1.0;

        let from2 = Square::make_square(Rank::Second, File::D);
        let to2 = Square::make_square(Rank::Fourth, File::D);
        let mv2 = ChessMove::new(from2, to2, None);
        let idx2 = move_to_policy_index(&mv2).unwrap();
        logits[idx2] = 0.0;

        let tensor = Tensor::from_slice(&logits, shape, &device).unwrap();
        let policy = convert_policy_tensor_to_action_probs(&tensor, &[mv1, mv2]).unwrap();

        let p1 = *policy.get(&ChessMove::new(from1, to1, None)).unwrap();
        let p2 = *policy.get(&ChessMove::new(from2, to2, None)).unwrap();

        // softmax([1,0])
        let e1 = 1.0f32.exp();
        let e0 = 1.0f32;
        let expected_p1 = e1 / (e1 + e0);
        let expected_p2 = e0 / (e1 + e0);
        approx_eq(p1, expected_p1, 1e-5);
        approx_eq(p2, expected_p2, 1e-5);
        approx_eq(p1 + p2, 1.0, 1e-5);

        // Ensure tensor is f32 to match inference.
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_encode_game_state_matches_add_state_for_current_plane_block() {
        let device = Device::Cpu;
        let state = ChessWrapper::new();
        let encoded = Chess::encode_game_state(&[state.clone()], &device);
        let flat = encoded.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // First block corresponds to current state piece planes.
        let mut expected = vec![0.0f32; BOARD_STATE_PLANES * 8 * 8];
        add_state(&mut expected, &state);

        assert_eq!(&flat[0..BOARD_STATE_PLANES * 8 * 8], expected.as_slice());

        // Standard planes are written at history_length() * planes.
        let standard_start = Chess::history_length() * BOARD_STATE_PLANES * 8 * 8;
        let plane_size = 8 * 8;
        let stm_plane = &flat[standard_start..standard_start + plane_size];
        assert!(stm_plane.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }
}
