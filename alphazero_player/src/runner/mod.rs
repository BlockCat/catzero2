use crate::{config::RunnerConfig, runner::messages::GamePlayed};
use alphazero_nn::AlphaGame;
use mcts::{AlphaZeroSelectionStrategy, DefaultAdjacencyTree, GameState, StateEvaluation, MCTS};
use std::fmt::Debug;
use tokio::{runtime::Runtime, sync::mpsc, task::JoinHandle};
use tokio_util::sync::CancellationToken;

#[cfg(feature = "chess")]
pub mod chess;

mod alpha_evaluator;
mod messages;

pub trait AlphaRunnable:
    AlphaGame<
    MoveType: Clone + Send + Sync + Debug + PartialEq + Eq,
    GameState: mcts::GameState<Action = Self::MoveType> + Send + Sync,
>
{
    type Config: Clone + Debug + Send + Sync + AlphaConfigurable;

    type StateEvaluation: StateEvaluation<<Self as AlphaGame>::GameState> + Send + Sync;

    fn create_evaluator(config: &Self::Config) -> Self::StateEvaluation;

    fn create_selector(config: &Self::Config) -> AlphaZeroSelectionStrategy;
}

pub trait AlphaConfigurable {
    fn discount_factor(&self) -> f32 {
        1.0
    }

    fn num_iterations(&self) -> usize {
        400
    }
    fn c1(&self) -> f32 {
        1.25
    }
    fn c2(&self) -> f32 {
        19652.0
    }
}

#[derive(Debug)]
pub enum RunnerError {
    GameError(anyhow::Error),
    Cancellation,
}

pub struct RunnerHandle {
    workers: Vec<JoinHandle<()>>,
    cancellation_token: CancellationToken,
}

/// RunnerService manages multiple game runners playing games in parallel.
pub struct RunnerService {
    config: RunnerConfig,
    rt: Runtime,
    handle: Option<RunnerHandle>,
    games_played: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl RunnerService {
    /// Creates a new RunnerService with the given configuration and inference service.
    pub fn new(config: RunnerConfig) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.threads)
            .enable_time()
            .build()
            .expect("Failed to create Tokio runtime");
        RunnerService {
            config,
            handle: None,
            rt,
            games_played: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Returns the total number of games played by this RunnerService in this session.
    pub fn games_played(&self) -> u64 {
        self.games_played.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns true if the RunnerService is currently running.
    pub fn is_running(&self) -> bool {
        self.handle.is_some()
    }

    /// Returns the number of games currently being played in parallel.
    pub fn games_playing(&self) -> usize {
        if self.is_running() {
            self.config.parallel_games
        } else {
            0
        }
    }

    /// Starts the RunnerService, spawning game runner tasks.
    pub fn start<G: AlphaRunnable + 'static>(
        &mut self,
        config: G::Config,
    ) -> Result<(), anyhow::Error> {
        if self.handle.is_some() {
            tracing::info!("RunnerService is already running");
            return Err(anyhow::anyhow!("RunnerService is already running"));
        }

        let cancellation_token = CancellationToken::new();

        let mut handles = Vec::with_capacity(self.config.parallel_games);

        let (game_tx, _game_rx) = mpsc::channel::<GamePlayed<G::GameState>>(100);

        for _ in 0..self.config.parallel_games {
            let cancellation_token = cancellation_token.clone();
            let game_tx = game_tx.clone();
            let games_played = self.games_played.clone();
            let config = config.clone();

            let handle = self.rt.spawn(async move {
                if let Err(e) =
                    start_runner::<G>(cancellation_token, config, game_tx, games_played).await
                {
                    match e {
                        RunnerError::Cancellation => {
                            tracing::info!("Runner task cancelled")
                        }
                        RunnerError::GameError(err) => {
                            tracing::error!("Runner task error: {:?}", err)
                        }
                    }
                }
            });

            handles.push(handle);
        }

        self.handle = Some(RunnerHandle {
            workers: handles,
            cancellation_token,
        });

        Ok(())

        // #[cfg(feature = "chess")]
        // {
        //     let handle = self.rt.spawn(async move {
        //         // Create a channel for game data collection

        //         use alphazero_chess::ChessWrapper;

        //         // Spawn a task to handle completed games
        //         let games_played_counter = games_played.clone();
        //         tokio::spawn(async move {
        //             while let Some(game) = game_rx.recv().await {
        //                 // Log game completion
        //                 tracing::info!("Game completed with {} moves", game.taken_actions.len());
        //                 games_played_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        //                 // TODO: Send game data to storage/training pipeline
        //             }
        //         });

        //         // Create the chess runner
        //         let chess_config = ChessConfig {
        //             num_iterations: config.num_iterations,
        //             discount_factor: 0.99,
        //             c1: 1.25,
        //             c2: 19652.0,
        //         };

        //         let runner = ChessRunner::new(chess_config, game_tx, inference_sender, device);

        //         RunnerService::start_async(config, rt, cancellation_token_clone, runner).await
        //     });

        //     self.handle = Some(RunnerHandle {
        //         task: handle,
        //         cancellation_token,
        //     });

        //     Ok(())
        // }
    }

    pub fn stop(&mut self) -> bool {
        match self.handle.take() {
            Some(handle) => {
                tracing::info!("Stopping RunnerService");
                drop(handle);
                true
            }
            _ => false,
        }
    }
}

/// Start a single game runner that plays games in a loop until cancellation.
async fn start_runner<G: AlphaRunnable + 'static>(
    cancellation_token: CancellationToken,
    game_config: G::Config,
    game_tx: mpsc::Sender<GamePlayed<G::GameState>>,
    games_played: std::sync::Arc<std::sync::atomic::AtomicU64>,
) -> Result<(), RunnerError> {
    while !cancellation_token.is_cancelled() {
        let game_played = play_a_game::<G>(&game_config, cancellation_token.clone()).await;

        games_played.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        match game_played {
            Ok(game) => {
                tracing::info!("Game completed with {} moves", game.taken_actions().len());
                let _ = game_tx.send(game).await;
            }
            Err(RunnerError::Cancellation) => {
                tracing::info!("Game runner received cancellation, exiting");
                break;
            }
            Err(RunnerError::GameError(e)) => {
                tracing::error!("Error during game play: {:?}", e);
            }
        }
    }

    Ok(())
}

async fn play_a_game<G: AlphaRunnable + 'static>(
    config: &G::Config,
    cancellation_token: CancellationToken,
) -> Result<GamePlayed<G::GameState>, RunnerError> {
    let selection_strategy = G::create_selector(config);
    let state_evaluation = G::create_evaluator(config);

    let mut state = G::GameState::default();
    let mut states = Vec::new();
    let mut policies = Vec::new();
    let mut taken_actions = Vec::new();

    let mut mcts: MCTS<
        G::GameState,
        G::MoveType,
        DefaultAdjacencyTree<G::MoveType>,
        AlphaZeroSelectionStrategy,
        G::StateEvaluation,
    > = MCTS::new(
        selection_strategy,
        state_evaluation,
        config.discount_factor(),
    )
    .map_err(|e| RunnerError::GameError(anyhow::anyhow!("Failed to create MCTS: {:?}", e)))?;

    while state.is_terminal().is_none() {
        if cancellation_token.is_cancelled() {
            return Err(RunnerError::Cancellation);
        }

        // Here you would use the inference_service to get policy and value predictions
        // and then select an action based on MCTS or other logic.

        // For demonstration, we will just take the first possible action.
        let possible_actions = state.get_possible_actions();
        if possible_actions.is_empty() {
            unimplemented!("No possible actions available");
        }

        let best_move = mcts
            .search_for_iterations_async(&state, config.num_iterations())
            .await
            .expect("MCTS search failed");

        states.push(state.clone());
        policies.push(mcts.get_action_probabilities());
        taken_actions.push(best_move.clone());

        state = state.take_action(best_move.clone());

        mcts.subtree_pruning(best_move).map_err(|e| {
            RunnerError::GameError(anyhow::anyhow!("Failed to prune MCTS subtree: {:?}", e))
        })?;
    }

    Err(RunnerError::Cancellation)
}

impl Drop for RunnerHandle {
    fn drop(&mut self) {
        tracing::info!("Runner stopping, cancelling tasks");
        self.cancellation_token.cancel();
        self.workers.iter().for_each(|s| s.abort());
    }
}

impl Drop for RunnerService {
    fn drop(&mut self) {
        tracing::info!("RunnerService stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default, Debug, PartialEq, Eq)]
    struct DummyState(isize);

    impl GameState for DummyState {
        type Action = isize;

        fn current_player_id(&self) -> usize {
            if self.0 % 2 == 0 {
                0
            } else {
                1
            }
        }

        fn get_possible_actions(&self) -> Vec<Self::Action> {
            vec![0, 1, 2]
        }

        fn take_action(&self, _action: Self::Action) -> Self {
            self.clone()
        }

        fn is_terminal(&self) -> Option<f32> {
            if self.0 >= 4 {
                Some(1.0)
            } else if self.0 <= -4 {
                Some(-1.0)
            } else {
                None
            }
        }
    }

    // impl SingleRunner for DummyState {
    //     type GameState = DummyState;

    //     fn tensor_input_shape(_historic_states: usize) -> Shape {
    //         Shape::from_dims(&[1])
    //     }

    //     fn play_game(
    //         &self,
    //         cancellation_token: CancellationToken,
    //     ) -> impl std::future::Future<Output = Result<GamePlayed<Self::GameState>, RunnerError>>
    //            + Send
    //            + Sync {
    //         let cancellation_token = cancellation_token.clone();
    //         async move {
    //             let mut state = self.clone();
    //             let mut states = Vec::new();
    //             let mut policies = Vec::new();
    //             let mut taken_actions = Vec::new();

    //             while cancellation_token.is_cancelled() == false {
    //                 states.push(state.clone());
    //                 let possible_actions = state.get_possible_actions();
    //                 let action = possible_actions[0];
    //                 let policy: Vec<(isize, f32)> = possible_actions
    //                     .iter()
    //                     .map(|&a| (a, if a == action { 1.0 } else { 0.0 }))
    //                     .collect();
    //                 policies.push(policy);
    //                 taken_actions.push(action);
    //                 state = state.take_action(action);

    //                 if let Some(_value) = state.is_terminal() {
    //                     break;
    //                 }
    //             }

    //             Ok(GamePlayed::new(
    //                 states,
    //                 policies,
    //                 taken_actions,
    //                 1.0,
    //                 Some(1),
    //             ))
    //         }
    //     }

    //     fn send_game_played(
    //         &self,
    //         _game_played: GamePlayed<Self::GameState>,
    //     ) -> impl std::future::Future<Output = Result<(), RunnerError>> + Send + Sync {
    //         async move { Ok(()) }
    //     }
    // }

    #[test]
    fn game_played_accessors_return_expected_data() {
        let service = RunnerService::new(Default::default());
        assert_eq!(service.games_played(), 0);
        assert_eq!(service.is_running(), false);
        assert_eq!(service.games_playing(), 0);
    }
}
