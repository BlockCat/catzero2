use crate::{
    config::InferenceConfig,
    error::{Error, Result},
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
    pub fn new(config: InferenceConfig, modus: Option<InferenceModusRequest>) -> Self {
        let modus = match modus {
            Some(InferenceModusRequest::SinglePlayer(model)) => {
                let worker = InferenceWorker::new(model, config.clone());
                InferenceModus::SinglePlayer(worker)
            }
            Some(InferenceModusRequest::Evaluator(models)) => {
                let workers = models
                    .into_iter()
                    .map(|model| InferenceWorker::new(Box::new(model), config.clone()))
                    .collect();
                InferenceModus::Evaluator(workers)
            }
            None => InferenceModus::None,
        };

        Self { modus }
    }

    pub fn update_model(&self, id: Option<usize>, weights: Vec<u8>) -> Result<()> {
        match &self.modus {
            InferenceModus::None => Err(Error::ModelLoadError(
                "InferenceService modus is None, cannot update model".to_string(),
            )),
            InferenceModus::SinglePlayer(inference_worker) => {
                inference_worker.update_model(weights)
            }

            InferenceModus::Evaluator(_inference_workers) => {
                if let Some(player_id) = id {
                    let inference_worker = _inference_workers
                        .get(player_id)
                        .ok_or_else(|| Error::ModelLoadError("Invalid player ID".to_string()))?;
                    inference_worker.update_model(weights)
                } else {
                    Err(Error::ModelLoadError(
                        "Must provide player ID to update model in Evaluator modus".to_string(),
                    ))
                }
            }
        }
    }

    pub async fn request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        match &self.modus {
            InferenceModus::None => Err(Error::InferenceError(
                "InferenceService modus is None".to_string(),
            )),
            InferenceModus::SinglePlayer(inference_worker) => {
                inference_worker.send_request(request).await
            }
            InferenceModus::Evaluator(inference_workers) => {
                let inference_worker = inference_workers
                    .get(request.player_id as usize)
                    .ok_or_else(|| Error::InferenceError("Invalid player ID".to_string()))?;

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
    pub fn new(model: Box<AlphaZeroNN>, config: InferenceConfig) -> Self {
        let (batch_service, worker_sender) = BatchService::new(config, model);
        let handle = batch_service.start();
        Self {
            worker_sender,
            _handle: handle,
        }
    }

    pub async fn send_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        let batcher_request = BatcherRequest {
            state_tensor: request.state_tensor,
            response_channel: response_tx,
        };
        self.worker_sender
            .send(batcher_request)
            .await
            .map_err(|e| {
                Error::InferenceError(format!("Failed to send request to batcher: {}", e))
            })?;
        let batch_response = response_rx.await.map_err(|e| {
            Error::InferenceError(format!("Failed to receive response from batcher: {}", e))
        })?;

        Ok(InferenceResponse {
            output_tensor: batch_response.output_tensor,
            value: batch_response.value,
        })
    }

    pub fn update_model(&self, _weights: Vec<u8>) -> Result<()> {
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
