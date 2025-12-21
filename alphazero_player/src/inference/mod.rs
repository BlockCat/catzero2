use crate::{
    config::BatcherConfig,
    inference::batcher::{BatchService, BatcherHandle, BatcherRequest},
};
use alphazero_nn::AlphaZeroNN;
use candle_core::Tensor;

mod batcher;

#[derive(Debug)]
pub struct InferenceService {
    modus: InferenceModus,
}

#[derive(Debug)]
enum InferenceModus {
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
                let worker = InferenceWorker::new(model, config.clone());
                InferenceModus::SinglePlayer(worker)
            }
            InferenceModusRequest::Evaluator(models) => {
                let workers = models
                    .into_iter()
                    .map(|model| InferenceWorker::new(Box::new(model), config.clone()))
                    .collect();
                InferenceModus::Evaluator(workers)
            }
        };

        Self { modus }
    }

    pub fn update_model(&self, id: Option<usize>, weights: Vec<u8>) -> Result<(), anyhow::Error> {
        match &self.modus {
            InferenceModus::None => Err(anyhow::anyhow!(
                "InferenceService modus is None, cannot update model"
            )),
            InferenceModus::SinglePlayer(inference_worker) => {
                inference_worker.update_model(weights)
            }
            InferenceModus::Evaluator(_inference_workers) => {
                if let Some(player_id) = id {
                    let inference_worker = _inference_workers
                        .get(player_id)
                        .ok_or_else(|| anyhow::anyhow!("Invalid player ID"))?;
                    inference_worker.update_model(weights)
                } else {
                    Err(anyhow::anyhow!(
                        "Must provide player ID to update model in Evaluator modus"
                    ))
                }
            }
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
    worker_sender: tokio::sync::mpsc::Sender<BatcherRequest>,
    _handle: BatcherHandle,
}

impl InferenceWorker {
    pub fn new(model: Box<AlphaZeroNN>, config: BatcherConfig) -> Self {
        let (batch_service, worker_sender) = BatchService::new(config, model);
        let handle = batch_service.start();
        Self {
            worker_sender,
            _handle: handle,
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

    pub fn update_model(&self, _weights: Vec<u8>) -> Result<(), anyhow::Error> {
        todo!()
        // self.worker_sender
        //     .try_send(BatcherRequest::UpdateModel { weights })
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
