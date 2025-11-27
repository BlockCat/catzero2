use crate::batcher::InferenceRequest;
use alphazero_chess::{chess::ChessMove, ChessWrapper};
use candle_core::Shape;
use mcts::GameState;
use tokio::{
    runtime::{Handle, Runtime},
    sync::mpsc,
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

mod chess;

pub trait SingleRunner {
    type GameState: GameState + Clone + Send + Sync + 'static;

    fn tensor_input_shape(historic_states: usize) -> Shape;

    fn play_game(
        &self,
        cancellation_token: CancellationToken,
    ) -> impl std::future::Future<Output = Result<GamePlayed<ChessWrapper>, RunnerError>> + Send + Sync;

    fn send_game_played(
        &self,
        game_played: GamePlayed<ChessWrapper>,
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
    pub states: Vec<G>,
    pub policies: Vec<Vec<(G::Action, f32)>>,
    pub taken_actions: Vec<G::Action>,
    pub value: f32,
    pub winner: Option<i32>,
}

unsafe impl<G: GameState + Send + Sync> Send for GamePlayed<G> {}
unsafe impl<G: GameState + Send + Sync> Sync for GamePlayed<G> {}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub num_iterations: usize,
    pub threads: usize,
    pub parallel_games: usize,
    pub inference_sender: mpsc::Sender<InferenceRequest>,
}

pub struct RunnerService {
    config: RunnerConfig,
    handle: Option<RunnerHandle>,
    rt: Runtime,
}

impl RunnerService {
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
        }
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
        let config = self.config.clone();
        let rt = self.rt.handle().clone();
        let cancellation_token = CancellationToken::new();
        let cancellation_token_clone = cancellation_token.clone();

        let handle = tokio::spawn(async move {
            RunnerService::start_async(config, rt, cancellation_token_clone).await
        });

        self.handle = Some(RunnerHandle {
            task: handle,
            cancellation_token,
        });

        Ok(())
    }

    pub fn stop(&mut self) -> bool {
        if let Some(handle) = self.handle.take() {
            tracing::info!("Stopping RunnerService");
            drop(handle);
            true
        } else {
            false
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
