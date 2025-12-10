use actix_web::{
    middleware::Logger,
    web::{scope, Data, ServiceConfig},
    App, HttpServer,
};
use alphazero_nn::{AlphaZeroNN, PolicyOutputType};
use candle_core::{DType, Device, Shape};
use candle_nn::{VarBuilder, VarMap};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{batcher::BatchService, config::ApplicationConfig};

mod api;
mod batcher;
mod runner;
mod model_repository;

mod config;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    // Load configuration from environment variables
    let config = ApplicationConfig::load();

    tracing::info!("Application configuration: {:?}", config);

    let device = Device::cuda_if_available(0).expect("Could not get device");

    let (model, _vb) = load_model(&device);

    tracing::info!("Model loaded on device: {:?}", device);
    
    _vb.save("./test.safetensors").expect("Could not save model");

    let (batcher, inference_sender) = BatchService::new(config.batcher_config.clone(), model);

    let batcher_handle = Arc::new(batcher.start());
    
    let runner_service = Arc::new(Mutex::new(runner::RunnerService::new(
        config.runner_config.clone(),
        inference_sender,
        device.clone(),
    )));

    let host = format!("{}:{}", config.host, config.port);
    let workers = config.server_workers;

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(Data::new(batcher_handle.clone()))
            .app_data(Data::new(config.clone()))
            .app_data(Data::new(runner_service.clone()))
            .service(scope("/api").configure(collect_routes))
    })
    .bind(host)?
    .workers(workers)    
    .run()
    .await?;

    Ok(())
}

fn collect_routes(cfg: &mut ServiceConfig) {
    cfg.service(api::status)
        .service(api::start_play)
        .service(api::stop_play)
        .service(api::update_model);
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
