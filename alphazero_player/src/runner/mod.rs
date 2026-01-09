use crate::{
    config::RunnerConfig,
    runner::messages::{GamePlayed, GameResult},
};
use alphazero_nn::AlphaGame;
use mcts::{
    AlphaZeroSelectionStrategy, DefaultAdjacencyTree, GameState, StateEvaluation, TerminalResult,
    MCTS,
};
use std::fmt::Debug;
use thiserror::Error;
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

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("Game error: {0}")]
    GameError(anyhow::Error),
    #[error("Operation was cancelled")]
    Cancellation,
    #[error("GamePlayed construction error: {0}")]
    GamePlayedError(String),
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
                        RunnerError::GamePlayedError(err) => {
                            tracing::error!("GamePlayed construction error: {}", err)
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
            Err(RunnerError::GamePlayedError(e)) => {
                tracing::error!("Error constructing GamePlayed: {}", e);
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

        let possible_actions = state.get_possible_actions();
        if possible_actions.is_empty() {
            return Err(RunnerError::GameError(anyhow::anyhow!(
                "No possible actions available in non-terminal state"
            )));
        }

        let best_move = mcts
            .search_for_iterations_async(&state, config.num_iterations())
            .await
            .map_err(|e| RunnerError::GameError(anyhow::anyhow!("MCTS search failed: {:?}", e)))?;

        states.push(state.clone());
        policies.push(mcts.get_action_probabilities());
        taken_actions.push(best_move.clone());

        state = state.take_action(best_move.clone());

        mcts.subtree_pruning(best_move).map_err(|e| {
            RunnerError::GameError(anyhow::anyhow!("Failed to prune MCTS subtree: {:?}", e))
        })?;
    }

    // With the assumption that it is two player zero-sum game
    let value = state
        .is_terminal()
        .unwrap()
        .to_player_perspective(0, state.current_player_id());

    let (value, winner) = match value {
        TerminalResult::Win => (1.0, GameResult::Winner(0)),
        TerminalResult::Loss => (-1.0, GameResult::Winner(1)),
        TerminalResult::Draw => (0.0, GameResult::Draw),
    };

    GamePlayed::new(states, policies, taken_actions, value, winner)
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

    #[test]
    fn game_played_accessors_return_expected_data() {
        let service = RunnerService::new(Default::default());
        assert_eq!(service.games_played(), 0);
        assert_eq!(service.is_running(), false);
        assert_eq!(service.games_playing(), 0);
    }
}
