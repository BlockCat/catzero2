use std::{collections::HashMap, sync::Arc};

use crate::{config::ApplicationConfig, inference::InferenceService, runner::RunnerService, Game};
use actix_web::{get, http::header::ContentType, post, *};
use candle_core::Device;
use tokio::sync::{Mutex, RwLock};

#[get("/status")]
async fn status(
    config: web::Data<ApplicationConfig>,
    runner_service: web::Data<Arc<Mutex<RunnerService>>>,
) -> ServerStatus {
    let runner_service = runner_service.lock().await;
    let play_info = if runner_service.is_running() {
        Some(PlayingInfo {
            threads: config.runner_config.threads,
            games_played: runner_service.games_played() as u32,
            games_playing: runner_service.games_playing() as u32,
            models: vec![],
        })
    } else {
        None
    };

    #[cfg(feature = "chess")]
    let game = Game::Chess;

    ServerStatus {
        running: true,
        game,
        playing: runner_service.is_running(),
        play_info,
        batch_info: BatchInfo {
            max_batch_size: config.inference_config.max_batch_size,
        },
    }
}

#[post("/play")]
async fn start_play(
    data: web::Data<Arc<Mutex<RunnerService>>>,
    inference_service: web::Data<Arc<RwLock<InferenceService>>>,
    device: web::Data<Device>,
    req: web::Json<PlayGameRequest>,
) -> HttpResponse {
    let mut service = data.lock().await;
    if service.is_running() {
        return HttpResponse::Ok().json(ServerMessage {
            message: "Play is already running".to_string(),
        });
    }

    let result = match req.game {
        Game::Chess => {
            #[cfg(feature = "chess")]
            {
                // service
                //     .start::<Chess>(ChessConfig {
                //         num_iterations: 300,
                //         device: Device::cuda_if_available(0).unwrap(),
                //         inference_service: inference_service.clone(),
                //         discount_factor: 1.0,
                //         c1: 1.25,
                //         c2: 19652.0,
                //     })
                //     .unwrap();

                use crate::runner::chess::{Chess, ChessConfig};

                service.start::<Chess>(ChessConfig {
                    num_iterations: req.num_iterations,
                    c1: req.c1,
                    c2: req.c2,
                    discount_factor: req.discount_factor,
                    inference_service: inference_service.get_ref().clone(),
                    device: device.get_ref().clone(),
                })
            }
            #[cfg(not(feature = "chess"))]
            {
                return HttpResponse::BadRequest().json(ServerMessage {
                    message: "Chess support is not enabled in this build.".to_string(),
                });
            }
        }
        _ => {
            return HttpResponse::BadRequest().json(ServerMessage {
                message: "Unsupported game".to_string(),
            });
        }
    };

    if let Err(e) = result {
        return HttpResponse::InternalServerError().json(ServerMessage {
            message: format!("Could not start play: {}", e),
        });
    }

    HttpResponse::Ok().json(ServerMessage {
        message: "Play started".to_string(),
    })
}

#[delete("/stop")]
async fn stop_play(data: web::Data<Arc<Mutex<RunnerService>>>) -> HttpResponse {
    let stopped = data.lock().await.stop();
    if stopped {
        HttpResponse::Ok().json(ServerMessage {
            message: "Play stopped".to_string(),
        })
    } else {
        HttpResponse::Ok().json(ServerMessage {
            message: "Play was not running".to_string(),
        })
    }
}

#[post("/update_model")]
async fn update_model(
    model_data: web::Json<ModelUpdateRequest>,
    inference_service: web::Data<Arc<RwLock<InferenceService>>>,
) -> HttpResponse {
    // TODO: Implement model hot-swapping
    // 1. Load new model weights from the provided path or data
    // 2. Update the batcher's model reference
    // 3. Optionally restart runner to use new model immediately
    let mut inference_service = inference_service.write().await;
    match &model_data.mode {
        ModelUpdateMode::SelfPlay { model_path } => {
            let result = inference_service
                .update_model(None, model_path, HashMap::new())
                .await;
        }
        ModelUpdateMode::Evaluate {
            best_model_path,
            challenger_model_path,
        } => {
            // Load challenger model weights and set up evaluation

            let result_best = inference_service
                .update_model(Some(true), best_model_path, HashMap::new())
                .await;
            let result_challenger = inference_service
                .update_model(Some(false), challenger_model_path, HashMap::new())
                .await;
        }
    };

    HttpResponse::Ok().json(ServerMessage {
        message: "Model update endpoint not yet implemented. This will support hot-swapping neural network weights.".to_string(),
    })
}

#[derive(serde::Deserialize)]
struct ModelUpdateRequest {
    mode: ModelUpdateMode,
}

#[derive(serde::Deserialize)]
#[serde(tag = "type")]
enum ModelUpdateMode {
    SelfPlay {
        model_path: String,
    },
    Evaluate {
        best_model_path: String,
        challenger_model_path: String,
    },
}

#[derive(serde::Serialize)]
struct ServerMessage {
    message: String,
}
#[derive(serde::Serialize)]
struct ServerStatus {
    running: bool,
    playing: bool,
    game: Game,
    play_info: Option<PlayingInfo>,
    batch_info: BatchInfo,
}

#[derive(serde::Serialize)]
struct PlayingInfo {
    threads: usize,
    games_played: u32,
    games_playing: u32,
    models: Vec<ModelInfo>,
}

#[derive(serde::Serialize)]
struct ModelInfo {
    model_name: String,
    hyper_params: Vec<HyperParamInfo>,
}

#[derive(serde::Serialize)]
struct HyperParamInfo {
    pub name: String,
    pub value: f32,
}

#[derive(serde::Serialize)]
struct BatchInfo {
    max_batch_size: usize,
}

impl Responder for ServerStatus {
    type Body = actix_web::body::BoxBody;

    fn respond_to(self, _req: &HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();

        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
}

#[derive(serde::Deserialize)]
struct PlayGameRequest {
    game: Game,
    num_iterations: usize,
    c1: f32,
    c2: f32,
    discount_factor: f32,
}
