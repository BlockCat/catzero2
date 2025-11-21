use actix::{Actor, Recipient};
use alphazero_chess::{
    chess::{ChessMove, Game},
    ChessWrapper,
};
use candle_core::{Device, Tensor};
use mcts::{
    AlphaZeroSelectionStrategy, DefaultAdjacencyTree, ModelEvaluation, StateEvaluation, MCTS,
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
}

impl ChessRunnerActor {
    pub fn new(
        c1: f32,
        c2: f32,
        discount_factor: f64,
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

        ChessRunnerActor { mcts }
    }
}

impl Actor for ChessRunnerActor {
    type Context = actix::Context<Self>;
}

struct ActorAlphaEvaluator {
    historic_moves: usize,
    batcher: Recipient<InferenceRequest>,
    device: Device,
}

impl StateEvaluation<ChessWrapper> for ActorAlphaEvaluator {
    async fn evaluation(&self, state: &ChessWrapper, previous_state: &[ChessWrapper]) -> ModelEvaluation {
        let standard_planes = 7;
        let board_plane = 14;
        let total_planes = standard_planes * self.historic_moves + board_plane;

        let tensor =
            Tensor::full(0.0, (total_planes, 8, 8), &self.device).expect("Could not create tensor");

        let player_id = state.0.side_to_move().to_index() as u32;
        let response = self.batcher
            .send(InferenceRequest {
                player_id,
                input: tensor,
            })
            .await
            .expect("Failed to send InferenceRequest");

        ModelEvaluation::new(vec![], response.value as f64)
    }
}
