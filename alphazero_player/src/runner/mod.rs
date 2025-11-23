use std::{mem::replace, sync::Arc};

use crate::batcher::InferenceRequest;
use tokio::{
    runtime::{Handle, Runtime},
    task::{JoinHandle, JoinSet, spawn_blocking},
};

pub struct RunnerHandle(JoinHandle<Result<(), anyhow::Error>>);

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

        let handle = tokio::spawn(async move { RunnerService::start_async(config, rt).await });

        let runner_handle = RunnerHandle(handle);
        self.handle = Some(runner_handle);

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

    pub async fn start_async(config: RunnerConfig, rt: Handle) -> Result<(), anyhow::Error> {
        tracing::info!("RunnerService starting");
        let num_iterations = config.num_iterations;
        let parallel_games = config.parallel_games;
        let channel = config.channel.clone();

        let mut handles = Vec::with_capacity(parallel_games);


        for i in 0..parallel_games {
            let x = channel.clone();
            let handle = rt.spawn(async move {
                let x = x;
                loop {
                    // Here would be the game logic using MCTS and the channel for inference requests
                    tracing::info!("Runner {} is playing a game...", i);

                    // Simulate some work
                    tokio::time::sleep(std::time::Duration::from_millis(i as u64 + 500)).await
                }
            });
            handles.push(handle);
        }


        for handle in handles {
            let _ = handle.await;
        }

        unreachable!()
    }
}

impl Drop for RunnerHandle {
    fn drop(&mut self) {
        tracing::info!("Runner stopping, quitting tasks");
        self.0.abort();
    }
}

impl Drop for RunnerService {
    fn drop(&mut self) {
        tracing::info!("RunnerService stopped");
        

    }
}
