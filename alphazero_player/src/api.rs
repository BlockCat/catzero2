
use actix::Addr;
use actix_web::{http::header::ContentType, *};

use crate::{
    actors::{
        batch_actor::{self, BatchActor},
        play_actor::{self, PlayActor, StartPlay, StopPlay},
    },
    Game,
};

#[get("/status")]
async fn status(
    data: web::Data<crate::AppState>,
    play_actor: web::Data<Addr<PlayActor>>,
    batch_actor: web::Data<Addr<BatchActor>>,
) -> ServerStatus {
    let play_response = play_actor
        .send(play_actor::InfoRequest)
        .await
        .expect("Could not get play info");
    let batch_response = batch_actor
        .send(batch_actor::InfoRequest)
        .await
        .expect("Could not get batch actor info");

    let play_info = if play_response.running {
        let models = Vec::new();

        

        Some(PlayingInfo {
            cpu_cores: play_response.cpu_cores,
            games_played: play_response.games_played,
            total_moves: 0,
            games_playing: 0,
            models,
        })
    } else {
        None
    };

    ServerStatus {
        running: true,
        game: data.game.clone(),
        playing: play_response.running,
        play_info,
        batch_info: BatchInfo {
            batch_size: batch_response.batch_size,
            max_wait_time_ms: batch_response.max_wait.as_millis() as u64,
        },
    }
}

#[get("/start_play")]
async fn start_play(data: web::Data<Addr<PlayActor>>) -> HttpResponse {
    data.send(StartPlay)
        .await
        .expect("Failed to send StartPlay message");
    HttpResponse::Ok().json(ServerMessage {
        message: "Play started".to_string(),
    })
}

#[get("/stop_play")]
async fn stop_play(data: web::Data<Addr<PlayActor>>) -> HttpResponse {
    data.send(StopPlay)
        .await
        .expect("Failed to send StopPlay message");
    HttpResponse::Ok().json(ServerMessage {
        message: "Play stopped".to_string(),
    })
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
    cpu_cores: usize,
    games_played: u32,
    total_moves: u32,
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
    batch_size: usize,
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
