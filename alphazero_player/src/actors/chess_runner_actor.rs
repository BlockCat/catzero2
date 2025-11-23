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
