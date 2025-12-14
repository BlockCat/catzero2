use std::sync::Arc;

use crate::{
    config::RunnerConfig,
    inference::InferenceService,
};
use candle_core::{Device, Shape};
use mcts::GameState;
use tokio::{
    runtime::{Handle, Runtime},
    sync::mpsc,
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

#[cfg(feature = "chess")]
pub mod chess;

#[cfg(feature = "chess")]
use chess::{ChessConfig, ChessRunner};

pub trait SingleRunner {
    type GameState: GameState + Clone + Send + Sync + 'static;

    fn tensor_input_shape(historic_states: usize) -> Shape;

    fn play_game(
        &self,
        cancellation_token: CancellationToken,
    ) -> impl std::future::Future<Output = Result<GamePlayed<Self::GameState>, RunnerError>> + Send + Sync;

    fn send_game_played(
        &self,
        game_played: GamePlayed<Self::GameState>,
    ) -> impl std::future::Future<Output = Result<(), RunnerError>> + Send + Sync;
}

#[derive(Debug)]
pub enum RunnerError {
    GameError(anyhow::Error),
    Cancellation,
}

pub struct RunnerHandle {
    task: JoinHandle<Result<(), anyhow::Error>>,
    cancellation_token: CancellationToken,
}

#[derive(Clone, Debug)]
pub struct GamePlayed<G: GameState + Send + Sync> {
    states: Vec<G>,
    policies: Vec<Vec<(G::Action, f32)>>,
    taken_actions: Vec<G::Action>,
    value: f32,
    winner: Option<i32>,
}

unsafe impl<G: GameState + Send + Sync> Send for GamePlayed<G> {}
unsafe impl<G: GameState + Send + Sync> Sync for GamePlayed<G> {}

impl<G: GameState + Send + Sync> GamePlayed<G> {
    pub fn new(
        states: Vec<G>,
        policies: Vec<Vec<(G::Action, f32)>>,
        taken_actions: Vec<G::Action>,
        value: f32,
        winner: Option<i32>,
    ) -> Self {
        GamePlayed {
            states,
            policies,
            taken_actions,
            value,
            winner,
        }
    }

    pub fn states(&self) -> &Vec<G> {
        &self.states
    }

    pub fn policies(&self) -> &Vec<Vec<(G::Action, f32)>> {
        &self.policies
    }

    pub fn taken_actions(&self) -> &Vec<G::Action> {
        &self.taken_actions
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn winner(&self) -> Option<i32> {
        self.winner
    }
}

pub struct RunnerService {
    config: RunnerConfig,
    handle: Option<RunnerHandle>,
    rt: Runtime,
    inference_sender: Arc<InferenceService>,
    device: Device,
    games_played: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl RunnerService {
    pub fn new(
        config: RunnerConfig,
        inference_sender: Arc<InferenceService>,
        device: Device,
    ) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.threads)
            .enable_time()
            .build()
            .expect("Failed to create Tokio runtime");
        RunnerService {
            config,
            handle: None,
            rt,
            inference_sender,
            device,
            games_played: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    pub fn games_played(&self) -> u64 {
        self.games_played.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn is_running(&self) -> bool {
        self.handle.is_some()
    }

    pub fn games_playing(&self) -> usize {
        if self.is_running() {
            self.config.parallel_games
        } else {
            0
        }
    }

    pub fn start(&mut self) -> Result<(), anyhow::Error> {
        if self.handle.is_some() {
            tracing::info!("RunnerService is already running");
            return Err(anyhow::anyhow!("RunnerService is already running"));
        }

        #[cfg(feature = "chess")]
        {
            let config = self.config.clone();
            let rt = self.rt.handle().clone();
            let cancellation_token = CancellationToken::new();
            let cancellation_token_clone = cancellation_token.clone();
            let inference_sender = self.inference_sender.clone();
            let device = self.device.clone();
            let games_played = self.games_played.clone();

            let handle = self.rt.spawn(async move {
                // Create a channel for game data collection

                use alphazero_chess::ChessWrapper;
                let (game_tx, mut game_rx) = mpsc::channel::<GamePlayed<ChessWrapper>>(100);

                // Spawn a task to handle completed games
                let games_played_counter = games_played.clone();
                tokio::spawn(async move {
                    while let Some(game) = game_rx.recv().await {
                        // Log game completion
                        tracing::info!("Game completed with {} moves", game.taken_actions.len());
                        games_played_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // TODO: Send game data to storage/training pipeline
                    }
                });

                // Create the chess runner
                let chess_config = ChessConfig {
                    num_iterations: config.num_iterations,
                    discount_factor: 0.99,
                    c1: 1.25,
                    c2: 19652.0,
                };

                let runner = ChessRunner::new(chess_config, game_tx, inference_sender, device);

                RunnerService::start_async(config, rt, cancellation_token_clone, runner).await
            });

            self.handle = Some(RunnerHandle {
                task: handle,
                cancellation_token,
            });

            Ok(())
        }

        #[cfg(not(feature = "chess"))]
        {
            Err(anyhow::anyhow!(
                "No game implementation enabled. Enable a feature like 'chess'"
            ))
        }
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

    pub async fn start_async(
        config: RunnerConfig,
        rt: Handle,
        cancellation_token: CancellationToken,
        service: impl SingleRunner + Send + Sync + Clone + 'static,
    ) -> Result<(), anyhow::Error> {
        tracing::info!("RunnerService starting");
        let _num_iterations = config.num_iterations;
        let parallel_games = config.parallel_games;

        let mut handles = Vec::with_capacity(parallel_games);

        for i in 0..parallel_games {
            let cancellation_token = cancellation_token.clone();
            let service = service.clone();

            let handle = rt.spawn(async move {
                'outer: loop {
                    if cancellation_token.is_cancelled() {
                        tracing::info!("Runner task {} received cancellation, exiting", i);
                        break 'outer;
                    }

                    let game_played = service
                        .play_game(cancellation_token.clone())
                        .await
                        .expect("Could not play game");

                    service
                        .send_game_played(game_played)
                        .await
                        .expect("Could not send game played");
                }
            });
            handles.push(handle);
        }

        // Wait for cancellation or for all tasks to complete
        wait_for_end_of_tasks(cancellation_token, handles).await;

        Ok(())
    }
}

async fn wait_for_end_of_tasks(
    cancellation_token: CancellationToken,
    handles: Vec<JoinHandle<()>>,
) {
    tokio::select! {
        _ = cancellation_token.cancelled() => {
            tracing::info!("RunnerService received cancellation, waiting for tasks to finish");
        }
        _ = async {
            for handle in handles {
                let _ = handle.await;
            }
        } => {
            tracing::info!("All runner tasks completed");
        }
    }
}

impl Drop for RunnerHandle {
    fn drop(&mut self) {
        tracing::info!("Runner stopping, cancelling tasks");
        self.cancellation_token.cancel();
        self.task.abort();
    }
}

impl Drop for RunnerService {
    fn drop(&mut self) {
        tracing::info!("RunnerService stopped");
    }
}
