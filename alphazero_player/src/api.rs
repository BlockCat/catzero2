use std::sync::Arc;

use crate::{
    batcher::BatcherConfig,
    runner::{RunnerConfig, RunnerService},
    Game,
};
use actix_web::{http::header::ContentType, *};
use tokio::sync::Mutex;

#[get("/status")]
async fn status(
    data: web::Data<crate::AppState>,
    batch_config: web::Data<BatcherConfig>,
    runner_config: web::Data<RunnerConfig>,
    runner_service: web::Data<Arc<Mutex<RunnerService>>>,
) -> ServerStatus {

    let runner_service = runner_service.lock().await;

    let play_info = if runner_service.is_running() {
        Some(PlayingInfo {
            threads: runner_config.threads,
            games_played: 0,
            games_playing: runner_service.games_playing() as u32,
            models: vec![],
        })
    } else {
        None
    };

    ServerStatus {
        running: true,
        game: data.game.clone(),
        playing: runner_service.is_running(),
        play_info,
        batch_info: BatchInfo {
            max_batch_size: batch_config.max_batch_size,
            min_batch_size: batch_config.min_batch_size,
            max_wait_time_ms: batch_config.max_wait.as_millis() as u64,
        },
    }
}

#[get("/start_play")]
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

#[get("/stop_play")]
async fn stop_play(data: web::Data<Arc<Mutex<RunnerService>>>) -> HttpResponse {
    let stopped = data.lock().await.stop();
    if stopped {
        HttpResponse::Ok().json(ServerMessage {
            message: "Play stopped".to_string(),
        })
    } else {
        return HttpResponse::Ok().json(ServerMessage {
            message: "Play was not running".to_string(),
        });
    }
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
    min_batch_size: usize,
    max_wait_time_ms: u64,
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
