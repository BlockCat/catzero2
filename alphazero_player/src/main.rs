use crate::{
    config::ApplicationConfig,
    error::{Error, Result},
    inference::InferenceService,
};
use actix_web::{
    middleware::Logger,
    web::{scope, Data, ServiceConfig},
    App, HttpServer,
};
use alphazero_nn::AlphaZeroNN;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use std::sync::Arc;
use tokio::sync::Mutex;

mod api;
mod config;
mod error;
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

    tracing::info!("Using device: {:?}", device);

    let (_model, _vb) =
        load_model(&device, config.alpha_zero_config.clone()).expect("Could not load model");

    tracing::info!("Model loaded");

    _vb.save("./test.safetensors")
        .expect("Could not save model");

    let inference_service = Arc::new(InferenceService::new(config.inference_config.clone(), None));

    // #[cfg(feature = "chess")]
    // runner_service
    //     .start::<Chess>(ChessConfig {
    //         num_iterations: 300,
    //         device: device.clone(),
    //         inference_service: inference_service.clone(),
    //         discount_factor: 1.0,
    //         c1: 1.25,
    //         c2: 19652.0,
    //     })
    //     .unwrap();

    let host = format!("{}:{}", config.host, config.port);
    let workers = config.server_workers;

    let runner_service = Arc::new(Mutex::new(runner::RunnerService::new(
        config.runner_config.clone(),
    )));

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum Game {
    Chess,
    PacoSaco,
    MatchThreeConnectFour,
}

fn load_model(
    device: &Device,
    alpha_zero_config: alphazero_nn::Config,
) -> Result<(AlphaZeroNN, VarMap)> {
    tracing::info_span!("load model").in_scope(|| {
        let (vb, varmap) = tracing::info_span!("create vars")
            .record("device", format!("{:?}", device))
            .in_scope(|| {
                let varmap = VarMap::new();
                let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
                (vb, varmap)
            });
        let model = tracing::info_span!("create model")
            .in_scope(|| AlphaZeroNN::new(alpha_zero_config, vb.clone()))
            .map_err(|e| Error::ModelLoadError(format!("Could not create model: {}", e)))?;
        Ok((model, varmap))
    })
}
