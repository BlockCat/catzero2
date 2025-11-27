use std::{env, sync::Arc};

use actix_web::{
    middleware::Logger,
    web::{scope, Data, ServiceConfig},
    App, HttpServer,
};
use alphazero_nn::{AlphaZeroNN, PolicyOutputType};
use candle_core::{DType, Device, Shape};
use candle_nn::{VarBuilder, VarMap};
use tokio::sync::Mutex;

use crate::{
    // actors::{batch_actor::BatchActor, play_actor::PlayActor},
    batcher::{BatchService, BatcherConfig},
};

// mod actors;
mod api;
mod batcher;
mod runner;

fn get_env_var(key: &str, default: &str) -> String {
    env::var(key).unwrap_or(default.to_string())
}
fn get_env_var_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .unwrap_or(default.to_string())
        .parse::<usize>()
        .expect(&format!("Could not parse {}", key))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    // Load configuration from environment variables
    let host = get_env_var("HOST", "127.0.0.1");
    let port = get_env_var("PORT", "8080");
    let num = get_env_var_usize("WORKERS", 4);

    tracing::info!("Starting server on {}:{} with {} workers", host, port, num);

    let state = AppState { game: Game::Chess };

    let device = Device::cuda_if_available(0).expect("Could not get device");

    let (model, _vb) = load_model(&device);
    let batch_config = create_batcher_config();

    let (batcher, inference_sender) = BatchService::new(batch_config.clone(), model);

    let runner_config = create_runner_config(inference_sender);

    tracing::info!("Batcher config: {:?}", batch_config);
    tracing::info!("Runner config: {:?}", runner_config);

    let runner_service = Arc::new(Mutex::new(runner::RunnerService::new(
        runner_config.clone(),
    )));

    let batcher_handle = Arc::new(batcher.start());

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(Data::new(batcher_handle.clone()))
            .app_data(Data::new(state.clone()))
            .app_data(Data::new(batch_config.clone()))
            .app_data(Data::new(runner_config.clone()))
            .app_data(Data::new(runner_service.clone()))
            .service(scope("/api").configure(collect_routes))
    })
    .bind(format!("{}:{}", host, port))?
    .workers(num)
    .run()
    .await?;

    Ok(())
}

fn create_batcher_config() -> BatcherConfig {
    let max_wait_ms = get_env_var_usize("MAX_WAIT_MS", 10);
    let max_batch_size = get_env_var_usize("MAX_BATCH_SIZE", 200);
    let min_batch_size = get_env_var_usize("MIN_BATCH_SIZE", 100);
    BatcherConfig {
        max_wait: std::time::Duration::from_millis(max_wait_ms as u64),
        max_batch_size,
        min_batch_size,
    }
}

fn create_runner_config(
    inference_sender: tokio::sync::mpsc::Sender<batcher::InferenceRequest>,
) -> runner::RunnerConfig {
    let threads = get_env_var_usize("PLAY_CORES", 6);
    let parallel_games = get_env_var_usize("PARALLEL_GAMES", 200);
    let num_iterations = get_env_var_usize("NUM_ITERATIONS", 400);
    runner::RunnerConfig {
        num_iterations,
        threads,
        parallel_games,
        inference_sender,
    }
}

fn collect_routes(cfg: &mut ServiceConfig) {
    cfg.service(api::status)
        .service(api::start_play)
        .service(api::stop_play);
}

#[derive(Clone, Debug)]
pub struct AppState {
    game: Game,
}

#[derive(Clone, Debug, serde::Serialize)]
enum Game {
    Chess,
    PacoSaco,
    MatchThreeConnectFour,
}

fn load_model(device: &Device) -> (AlphaZeroNN, VarMap) {
    tracing::info_span!("load model").in_scope(|| {
        let (vb, varmap) = tracing::info_span!("create vars")
            .record("device", format!("{:?}", device))
            .in_scope(|| {
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
                (vb, varmap)
            });
        let model = tracing::info_span!("create model").in_scope(|| {
            AlphaZeroNN::new(
                alphazero_nn::Config {
                    n_input_channels: 17,
                    n_residual_blocks: 10,
                    n_filters: 256,
                    kernel_size: 3,
                    game_shape: Shape::from_dims(&[8, 8]),
                    policy_output_type: PolicyOutputType::Flat(4672),
                },
                vb.clone(),
            )
            .expect("Could not create model")
        });
        (model, varmap)
    })
}
