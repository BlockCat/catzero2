use alphazero_nn::AlphaZeroNN;
use candle_core::Tensor;

use crate::{
    config::BatcherConfig,
    inference::batcher::{BatchService, BatcherHandle, BatcherRequest},
};

mod batcher;

#[derive(Debug)]
pub struct InferenceService {
    config: BatcherConfig,
    modus: InferenceModus,
}

#[derive(Debug)]
pub enum InferenceModus {
    None,
    SinglePlayer(InferenceWorker),
    Evaluator(Vec<InferenceWorker>),
}

#[derive(Debug)]
pub enum InferenceModusRequest {
    SinglePlayer(Box<AlphaZeroNN>),
    Evaluator(Vec<AlphaZeroNN>),
}

impl InferenceService {
    pub fn new(config: BatcherConfig, modus: InferenceModusRequest) -> Self {

        let modus = match modus {
            InferenceModusRequest::SinglePlayer(model) => {
                let worker = InferenceWorker::new(0, model, config.clone());
                InferenceModus::SinglePlayer(worker)
            }
            InferenceModusRequest::Evaluator(models) => {
                let workers = models
                    .into_iter()
                    .enumerate()
                    .map(|(i, model)| InferenceWorker::new(i as u32, Box::new(model), config.clone()))
                    .collect();
                InferenceModus::Evaluator(workers)
            }
        };

        Self {
            config,
            modus,
        }
    }

    pub async fn request(&self, request: InferenceRequest) -> InferenceResponse {
        match &self.modus {
            InferenceModus::None => {
                panic!("InferenceService modus is None");
            }
            InferenceModus::SinglePlayer(inference_worker) => {
                inference_worker.send_request(request).await
            }
            InferenceModus::Evaluator(inference_workers) => {
                let inference_worker = inference_workers
                    .get(request.player_id as usize)
                    .expect("Invalid player ID");
                inference_worker.send_request(request).await
            }
        }
    }

}


#[derive(Debug)]
struct InferenceWorker {
    id: u32,
    worker_sender: tokio::sync::mpsc::Sender<BatcherRequest>,
    handle: BatcherHandle,
}

impl InferenceWorker {
    pub fn new(id: u32, model: Box<AlphaZeroNN>, config: BatcherConfig) -> Self {
        let (batch_service, worker_sender) = BatchService::new(config, model);
        let handle = batch_service.start();
        Self {
            id,
            worker_sender,
            handle,
        }
    }

    pub async fn send_request(&self, request: InferenceRequest) -> InferenceResponse {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        let batcher_request = BatcherRequest {
            state_tensor: request.state_tensor,
            response_channel: response_tx,
        };
        self.worker_sender
            .send(batcher_request)
            .await
            .expect("Could not send request to batcher");
        let batch_response = response_rx
            .await
            .expect("Could not receive response from batcher");

        InferenceResponse {
            output_tensor: batch_response.output_tensor,
            value: batch_response.value,
        }
    }
}

#[derive(Debug)]
pub struct InferenceRequest {
    pub player_id: u32,
    pub state_tensor: Tensor,
}

#[derive(Clone, Debug)]
pub struct InferenceResponse {
    pub output_tensor: Tensor,
    pub value: f32,
}
