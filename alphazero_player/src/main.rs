use std::env;

use actix::Actor;
use actix_web::{
    middleware::Logger,
    web::{scope, Data, ServiceConfig},
    App, HttpServer,
};

use crate::actors::{batch_actor::BatchActor, play_actor::PlayActor};

mod actors;
mod api;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    // Load configuration from environment variables
    let host = env::var("HOST").unwrap_or("127.0.0.1".to_string());
    let port = env::var("PORT").unwrap_or("8080".to_string());
    let num = env::var("WORKERS")
        .unwrap_or("4".to_string())
        .parse::<usize>()
        .expect("Could not parse WORKERS");

    let play_cores = env::var("PLAY_CORES")
        .unwrap_or("6".to_string())
        .parse::<usize>()
        .expect("Could not parse PLAY_CORES");
    let parallel_games = env::var("PARALLEL_GAMES")
        .unwrap_or("300".to_string())
        .parse::<usize>()
        .expect("Could not parse PARALLEL_GAMES");

    let state = AppState {
        game: Game::Chess,
        play_cores,
    };

    let batch_actor_addr = BatchActor::new().start();
    let play_actor_addr = PlayActor::new(
        play_cores,
        parallel_games,
        batch_actor_addr.clone().recipient(),
    )
    .start();

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default())
            .app_data(Data::new(state.clone()))
            .app_data(Data::new(batch_actor_addr.clone()))
            .app_data(Data::new(play_actor_addr.clone()))
            .service(scope("/api").configure(collect_routes))
    })
    .bind(format!("{}:{}", host, port))?
    .workers(num)
    .run()
    .await
}

fn collect_routes(cfg: &mut ServiceConfig) {
    cfg.service(api::status)
        .service(api::start_play)
        .service(api::stop_play);
}

#[derive(Clone, Debug)]
pub struct AppState {
    game: Game,
    play_cores: usize,
}

#[derive(Clone, Debug, serde::Serialize)]
enum Game {
    Chess,
}
