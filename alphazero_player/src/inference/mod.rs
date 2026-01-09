use std::{collections::HashMap, fmt::Debug};

use crate::{
    config::InferenceConfig,
    error::{Error, Result},
    inference::batcher::{BatchService, BatcherHandle, BatcherRequest},
};
use alphazero_nn::AlphaZeroNN;
use candle_core::Tensor;
use candle_nn::VarMap;
use tokio::sync::{Mutex, RwLock};

mod batcher;

#[derive(Debug)]
pub struct InferenceService {
    modus: InferenceModus,
}

#[derive(Debug)]
enum InferenceModus {
    None,
    SinglePlayer(InferenceWorker),
    Evaluator {
        current_best_model: InferenceWorker,
        challenger_model: InferenceWorker,
    },
}

pub enum InferenceModusRequest {
    SinglePlayer {
        name: String,
        model: Box<AlphaZeroNN>,
        varmap: VarMap,
    },
    Evaluator {
        current_best_name: String,
        current_best_model: Box<AlphaZeroNN>,
        current_best_varmap: VarMap,
        challenger_name: String,
        challenger_model: Box<AlphaZeroNN>,
        challenger_varmap: VarMap,
    },
}

impl Debug for InferenceModusRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceModusRequest::SinglePlayer { name, model, .. } => f
                .debug_struct("InferenceModusRequest::SinglePlayer")
                .field("name", name)
                .field("model", model)
                .finish(),
            InferenceModusRequest::Evaluator {
                current_best_name,
                current_best_model,
                challenger_name,
                challenger_model,
                ..
            } => f
                .debug_struct("InferenceModusRequest::Evaluator")
                .field("current_best_name", current_best_name)
                .field("current_best_model", current_best_model)
                .field("challenger_name", challenger_name)
                .field("challenger_model", challenger_model)
                .finish(),
        }
    }
}

impl InferenceService {
    pub fn new(config: InferenceConfig, modus: Option<InferenceModusRequest>) -> Self {
        let modus = match modus {
            Some(InferenceModusRequest::SinglePlayer {
                name,
                model,
                varmap,
            }) => {
                let worker = InferenceWorker::new_with_name(name, model, &varmap, config.clone());
                InferenceModus::SinglePlayer(worker)
            }
            Some(InferenceModusRequest::Evaluator {
                current_best_name,
                current_best_model,
                current_best_varmap,
                challenger_name,
                challenger_model,
                challenger_varmap,
            }) => {
                let current_best_worker = InferenceWorker::new_with_name(
                    current_best_name,
                    current_best_model,
                    &current_best_varmap,
                    config.clone(),
                );
                let challenger_worker = InferenceWorker::new_with_name(
                    challenger_name,
                    challenger_model,
                    &challenger_varmap,
                    config.clone(),
                );
                InferenceModus::Evaluator {
                    current_best_model: current_best_worker,
                    challenger_model: challenger_worker,
                }
            }
            None => InferenceModus::None,
        };

        Self { modus }
    }

    pub async fn update_model(
        &mut self,
        best: Option<bool>,
        model_name: &str,
        weights: HashMap<String, Tensor>,
    ) -> Result<()> {
        match &mut self.modus {
            InferenceModus::None => Err(Error::ModelLoadError(
                "InferenceService modus is None, cannot update model".to_string(),
            )),
            InferenceModus::SinglePlayer(inference_worker) => {
                inference_worker.update_model(model_name, weights).await
            }

            InferenceModus::Evaluator {
                current_best_model,
                challenger_model: challenger,
            } => match best {
                Some(true) => current_best_model.update_model(model_name, weights).await,
                Some(false) => challenger.update_model(model_name, weights).await,
                None => Err(Error::ModelLoadError(
                    "Must specify if model is best or challenger".to_string(),
                )),
            },
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
            InferenceModus::Evaluator {
                current_best_model,
                challenger_model,
            } => {
                if request.challenger {
                    challenger_model.send_request(request).await
                } else {
                    current_best_model.send_request(request).await
                }
            }
        }
    }

    pub fn has_models(&self) -> bool {
        match &self.modus {
            InferenceModus::None => false,
            _ => true,
        }
    }
}

struct InferenceWorker {
    worker_sender: tokio::sync::mpsc::Sender<BatcherRequest>,
    varmap: VarMap,
    model_name: String,
    _handle: BatcherHandle,
}

impl Debug for InferenceWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceWorker")
            .field("model", &self.model_name)
            .field("worker_sender", &self.worker_sender)
            .finish()
    }
}

impl InferenceWorker {
    pub fn new(model: Box<AlphaZeroNN>, varmap: &VarMap, config: InferenceConfig) -> Self {
        Self::new_with_name("empty".to_string(), model, varmap, config)
    }

    pub fn new_with_name(
        model_name: String,
        model: Box<AlphaZeroNN>,
        varmap: &VarMap,
        config: InferenceConfig,
    ) -> Self {
        let (batch_service, worker_sender) = BatchService::new(config, model);
        let handle = batch_service.start();

        Self {
            worker_sender,
            varmap: varmap.clone(),
            model_name,
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

    pub async fn update_model(
        &mut self,
        model: &str,
        weights: HashMap<String, Tensor>,
    ) -> Result<()> {
        if self.model_name == model {
            return Ok(());
        }
        let varmap = &mut self.varmap;
        varmap.set(weights.into_iter()).map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed to load weights into VarMap for model {}: {}",
                model, e
            ))
        })?;
        self.model_name = model.to_string();
        Ok(())
    }
}

#[derive(Debug)]
pub struct InferenceRequest {
    pub challenger: bool,
    pub state_tensor: Tensor,
}

#[derive(Clone, Debug)]
pub struct InferenceResponse {
    pub output_tensor: Tensor,
    pub value: f32,
}
