use candle_core::Device;
use mcts::{GameState, ModelEvaluation, StateEvaluation};
use std::sync::Arc;

use crate::{
    inference::{InferenceRequest, InferenceService},
    runner::AlphaRunnable,
};

pub struct AlphaEvaluator<G: AlphaRunnable> {
    batcher: Arc<InferenceService>,
    device: Device,
    _marker: std::marker::PhantomData<G>,
}

impl<G: AlphaRunnable> AlphaEvaluator<G> {
    pub fn new(batcher: Arc<InferenceService>, device: Device) -> Self {
        Self {
            batcher,
            device,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<G: AlphaRunnable + Send + Sync> StateEvaluation<G::GameState> for AlphaEvaluator<G> {
    async fn evaluation(
        &self,
        state: &G::GameState,
        previous_state: &[G::GameState],
    ) -> ModelEvaluation<G::MoveType> {
        let mut states = Vec::with_capacity(previous_state.len() + 1);
        states.extend_from_slice(previous_state);
        states.push(state.clone());

        let tensor = G::encode_game_state(&states, &self.device);

        let response = self
            .batcher
            .request(InferenceRequest {
                player_id: state.current_player_id() as u32,
                state_tensor: tensor,
            })
            .await;

        let moves = state.get_possible_actions();

        let policy = G::decode_policy_tensor(&response.output_tensor, &moves)
            .expect("Failed to decode policy tensor");

        ModelEvaluation::new(policy, response.value)
    }
}
