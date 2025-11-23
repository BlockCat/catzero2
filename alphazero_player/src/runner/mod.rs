use crate::batcher::InferenceRequest;
use tokio::{
    runtime::{Handle, Runtime},
    task::JoinHandle,
};
use tokio_util::sync::CancellationToken;

pub struct RunnerHandle {
    task: JoinHandle<Result<(), anyhow::Error>>,
    cancellation_token: CancellationToken,
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub num_iterations: usize,
    pub threads: usize,
    pub parallel_games: usize,
    pub channel: tokio::sync::mpsc::Sender<InferenceRequest>,
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
                loop {
                    // Check for cancellation
                    if cancellation_token.is_cancelled() {
                        tracing::info!("Runner {} received cancellation signal, stopping...", i);
                        break;
                    }

                    // Here would be the game logic using MCTS and the channel for inference requests
                    tracing::info!("Runner {} is playing a game...", i);

                    // Simulate some work
                    tokio::time::sleep(std::time::Duration::from_millis(i as u64 + 500)).await
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

impl Drop for RunnerService {
    fn drop(&mut self) {
        tracing::info!("RunnerService stopped");
    }
}
