use crate::batcher::InferenceRequest;
use mcts::GameState;
use tokio::{
    runtime::{Handle, Runtime},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

mod chess;

pub trait SingleRunner {
    type GameState: GameState + Clone + Send + Sync + 'static;

    async fn play_game(
        &self,
        cancellation_token: CancellationToken,
    ) -> Result<GamePlayed<Self::GameState>, RunnerError>;
}

pub enum RunnerError {
    GameError(anyhow::Error),
    Cancellation,
}

pub struct RunnerHandle {
    task: JoinHandle<Result<(), anyhow::Error>>,
    cancellation_token: CancellationToken,
}

#[derive(Clone, Debug)]
pub struct GamePlayed<A: GameState> {
    pub states: Vec<A>,
    pub policies: Vec<Vec<(A::Action, f32)>>,
    pub taken_actions: Vec<A::Action>,
    pub value: f32,
    pub winner: Option<i32>,
}

unsafe impl <A: GameState + Send> Send for GamePlayed<A> {}
unsafe impl <A: GameState + Sync> Sync for GamePlayed<A> {}

#[derive(Debug, Clone)]
pub struct RunnerConfig<A: GameState + Clone> {
    pub num_iterations: usize,
    pub threads: usize,
    pub parallel_games: usize,
    pub channel: tokio::sync::mpsc::Sender<InferenceRequest>,
    pub game_played_channel: tokio::sync::mpsc::Sender<GamePlayed<A>>,
}

pub struct RunnerService<A: GameState + Clone> {
    config: RunnerConfig<A>,
    handle: Option<RunnerHandle>,
    rt: Runtime,
}

impl<A: GameState + Clone + Send + Sync + 'static> RunnerService<A> {
    pub fn new(config: RunnerConfig<A>) -> Self {
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
        config: RunnerConfig<A>,
        rt: Handle,
        cancellation_token: CancellationToken,
    ) -> Result<(), anyhow::Error> {
        tracing::info!("RunnerService starting");
        let _num_iterations = config.num_iterations;
        let parallel_games = config.parallel_games;
        let channel = config.channel.clone();

        let mut handles = Vec::with_capacity(parallel_games);

        for i in 0..parallel_games {
            let _channel = channel.clone();
            let cancellation_token = cancellation_token.clone();

            let handle = rt.spawn(async move {
                'outer: loop {
                    if cancellation_token.is_cancelled() {
                        tracing::info!("Runner task {} received cancellation, exiting", i);
                        break 'outer;
                    }
                  
                }
            });
            handles.push(handle);
        }

        // Wait for cancellation or for all tasks to complete
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

        Ok(())
    }
}

impl Drop for RunnerHandle {
    fn drop(&mut self) {
        tracing::info!("Runner stopping, cancelling tasks");
        self.cancellation_token.cancel();
        self.task.abort();
    }
}

impl<A: GameState + Clone> Drop for RunnerService<A> {
    fn drop(&mut self) {
        tracing::info!("RunnerService stopped");
    }
}
