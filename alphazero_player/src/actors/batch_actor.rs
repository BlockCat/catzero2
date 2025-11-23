use actix::{Actor, Context, Handler};
use std::time::Duration;
use tracing::{info, trace};

pub use messages::{InferenceRequest, InfoRequest, InfoResponse};

use crate::actors::batch_actor::messages::InferenceResponse;

#[derive(Debug)]
pub struct Model {
    pub name: String,
    pub version: String,
}

#[derive(Debug)]
pub enum PlayMode {
    None,
    SelfPlay(Model),
    Evaluation { white: Model, black: Model },
}

pub struct BatchActor {
    max_wait: Duration,
    batch_size: usize,
    play_mode: PlayMode,
}

impl BatchActor {
    pub fn new() -> Self {
        BatchActor {
            max_wait: Duration::from_millis(200),
            batch_size: 100,
            play_mode: PlayMode::None,
        }
    }

    pub fn with_max_wait(mut self, duration: Duration) -> Self {
        self.max_wait = duration;
        self
    }

    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.batch_size = max_size;
        self
    }

    pub fn with_play_mode(mut self, play_mode: PlayMode) -> Self {
        self.play_mode = play_mode;
        self
    }
}

impl Actor for BatchActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("BatchActor started with max_wait: {:?}", self.max_wait);
        info!("BatchActor started with play_mode: {:?}", self.play_mode);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("BatchActor stopped");
    }
}

impl Handler<messages::SwitchPlayMode> for BatchActor {
    type Result = ();

    fn handle(&mut self, msg: messages::SwitchPlayMode, _ctx: &mut Self::Context) -> Self::Result {
        info!("Switching PlayMode to: {:?}", msg.0);
        self.play_mode = msg.0;
    }
}

impl Handler<messages::InferenceRequest> for BatchActor {
    type Result = InferenceResponse;

    fn handle(
        &mut self,
        msg: messages::InferenceRequest,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        trace!(
            "Received InferenceRequest from player_id: {} with input shape: {:?}",
            msg.player_id,
            msg.input.shape()
        );

        // Placeholder: Echo the input as output
        InferenceResponse {
            output: msg.input,
            value: 0.0,
        }
    }
}

impl Handler<messages::InfoRequest> for BatchActor {
    type Result = InfoResponse;

    fn handle(&mut self, _msg: messages::InfoRequest, _ctx: &mut Self::Context) -> Self::Result {
        info!("Current PlayMode: {:?}", self.play_mode);
        InfoResponse {
            max_wait: self.max_wait,
            batch_size: self.batch_size,
        }
    }
}

mod messages {
    use actix::{
        dev::{MessageResponse, OneshotSender},
        Actor, Message,
    };
    use candle_core::Tensor;

    #[derive(Message)]
    #[rtype(result = "()")]
    pub struct SwitchPlayMode(pub super::PlayMode);

    #[derive(Message)]
    #[rtype(result = "InferenceResponse")]
    pub struct InferenceRequest {
        pub player_id: u32,
        pub input: Tensor,
    }

    pub struct InferenceResponse {
        pub output: Tensor,
        pub value: f32,
    }

    impl<A, M> MessageResponse<A, M> for InferenceResponse
    where
        A: Actor,
        M: Message<Result = InferenceResponse>,
    {
        fn handle(self, _ctx: &mut A::Context, tx: Option<OneshotSender<M::Result>>) {
            if let Some(tx) = tx {
                let _ = tx.send(self);
            }
        }
    }

    #[derive(Message)]
    #[rtype(result = "InfoResponse")]
    pub struct InfoRequest;

    pub struct InfoResponse {
        pub max_wait: std::time::Duration,
        pub batch_size: usize,
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
