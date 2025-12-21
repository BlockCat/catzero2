#[cfg(feature = "chess")]
use crate::runner::chess::{Chess, ChessConfig};
use crate::{config::ApplicationConfig, inference::InferenceService};
use actix_web::{
    middleware::Logger,
    web::{scope, Data, ServiceConfig},
    App, HttpServer,
};
use alphazero_nn::{AlphaZeroNN, PolicyOutputType};
use candle_core::{DType, Device, Shape};
use candle_nn::{VarBuilder, VarMap};
use std::sync::Arc;

mod api;
mod config;
mod inference;
mod model_repository;
mod runner;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    // Load configuration from environment variables
    let config = ApplicationConfig::load();

    tracing::info!("Application configuration: {:?}", config);

    let device = Device::cuda_if_available(0).expect("Could not get device");

    let (_model, _vb) = load_model(&device);

    tracing::info!("Model loaded on device: {:?}", device);

    _vb.save("./test.safetensors")
        .expect("Could not save model");

    // TODO: inference service really should be owned by a runner? And the runner service by the API server?
    // TODO: rethink ownership model here, but I think that we need to have an inference service per runner service at least,
    // so that we can stop/start them independently on a different modus.
    let inference_service = Arc::new(InferenceService::new(
        config.batcher_config.clone(),
        inference::InferenceModusRequest::Evaluator(vec![]),
    ));

    let mut runner_service = runner::RunnerService::new(config.runner_config.clone());

    #[cfg(feature = "chess")]
    runner_service
        .start::<Chess>(ChessConfig {
            num_iterations: 300,
            device: device.clone(),
            inference_service: inference_service.clone(),
            discount_factor: 1.0,
            c1: 1.25,
            c2: 19652.0,
        })
        .unwrap();

    let host = format!("{}:{}", config.host, config.port);
    let workers = config.server_workers;

    let runner_service = Arc::new(runner_service);

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(Data::new(inference_service.clone()))
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
