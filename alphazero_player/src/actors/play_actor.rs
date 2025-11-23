
use actix::{Actor, AsyncContext, Context, Handler, Recipient};
use tracing::info;

pub use messages::{InfoRequest, InfoResponse, StartPlay, StopPlay};

use crate::actors::{batch_actor::InferenceRequest, chess_runner_actor::ChessRunnerActor};

pub struct PlayActor {
    num_cores: usize,
    parallel_games: usize,
    fut: Option<actix::SpawnHandle>,
    batcher: Recipient<InferenceRequest>,
}

impl PlayActor {
    pub fn new(
        num_cores: usize,
        parallel_games: usize,
        batcher: Recipient<InferenceRequest>,
    ) -> Self {
        PlayActor {
            num_cores,
            parallel_games,
            fut: None,
            batcher,
        }
    }
}

impl Actor for PlayActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("PlayActor started with {} cores", self.num_cores);
    }
}

impl Handler<messages::StartPlay> for PlayActor {
    type Result = ();

    fn handle(&mut self, _msg: messages::StartPlay, _ctx: &mut Self::Context) -> Self::Result {
        if self.fut.is_some() {
            info!("Play is already running");
            return;
        }

        info!("Starting play with {} cores", self.num_cores);
        let _num_cores = self.num_cores;
        let parallel_games = self.parallel_games;

        let device =
            candle_core::Device::cuda_if_available(0).expect("Could not get inference device");

        for _id in 0..parallel_games {
            let _addr =
                ChessRunnerActor::new(1.0, 2.0, 1.0, self.batcher.clone(), device.clone()).start();
        }
    }
}

impl Handler<messages::StopPlay> for PlayActor {
    type Result = ();

    fn handle(&mut self, _msg: messages::StopPlay, ctx: &mut Self::Context) -> Self::Result {
        if let Some(handle) = self.fut.take() {
            info!("Stopping play");
            ctx.cancel_future(handle);
        } else {
            info!("Play is not running");
        }
    }
}

impl Handler<messages::InfoRequest> for PlayActor {
    type Result = InfoResponse;

    fn handle(&mut self, _msg: messages::InfoRequest, _ctx: &mut Self::Context) -> Self::Result {
        InfoResponse {
            running: self.fut.is_some(),
            cpu_cores: self.num_cores,
            games_played: 0, // Placeholder for actual games played count
        }
    }
}

mod messages {

    use actix::{
        dev::{MessageResponse, OneshotSender},
        Actor, Message,
    };
    #[derive(Message)]
    #[rtype(result = "()")]
    pub struct StartPlay;

    #[derive(Message)]
    #[rtype(result = "()")]
    pub struct StopPlay;

    #[derive(Message)]
    #[rtype(result = "InfoResponse")]
    pub struct InfoRequest;

    pub struct InfoResponse {
        pub running: bool,
        pub cpu_cores: usize,
        pub games_played: u32,
    }

    impl<A, M> MessageResponse<A, M> for InfoResponse
    where
        A: Actor,
        M: Message<Result = InfoResponse>,
    {
        fn handle(self, _ctx: &mut A::Context, tx: Option<OneshotSender<M::Result>>) {
            if let Some(tx) = tx {
                let _ = tx.send(self);
            }
        }
    }
}
