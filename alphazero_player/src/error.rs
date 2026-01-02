pub type Result<T> = std::result::Result<T, Error>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Model loading error: {0}")]
    ModelLoadError(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("Runner error: {0}")]
    RunnerError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Unknown error: {0}")]
    Unknown(String),
}
