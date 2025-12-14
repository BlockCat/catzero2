use std::sync::Arc;

use crate::{config::ApplicationConfig, runner::RunnerService, Game};
use actix_web::{get, http::header::ContentType, post, *};
use tokio::sync::Mutex;

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

    ServerStatus {
        running: true,
        game: Game::Chess,
        playing: runner_service.is_running(),
        play_info,
        batch_info: BatchInfo {
            max_batch_size: config.batcher_config.max_batch_size,
        },
    }
}

#[get("/play")]
async fn start_play(data: web::Data<Arc<Mutex<RunnerService>>>) -> HttpResponse {
    let result = data.lock().await.start();
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
async fn update_model(_model_data: web::Json<ModelUpdateRequest>) -> HttpResponse {
    // TODO: Implement model hot-swapping
    // 1. Load new model weights from the provided path or data
    // 2. Update the batcher's model reference
    // 3. Optionally restart runner to use new model immediately

    HttpResponse::Ok().json(ServerMessage {
        message: "Model update endpoint not yet implemented. This will support hot-swapping neural network weights.".to_string(),
    })
}

#[derive(serde::Deserialize)]
struct ModelUpdateRequest {
    #[allow(dead_code)]
    model_path: Option<String>,
    #[allow(dead_code)]
    model_data: Option<Vec<u8>>,
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
