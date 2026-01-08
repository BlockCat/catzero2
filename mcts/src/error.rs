use thiserror::Error;

pub type Result<T> = std::result::Result<T, MCTSError>;

#[derive(Debug, Error)]
pub enum MCTSError {
    #[error("The requested move was not found in the current game state.")]
    MoveNotFound,

    #[error("Tree node initialization failed.")]
    TreeNodeInitializationFailed,

    #[error("Error during node expansion in the MCTS tree.")]
    ExpansionError,

    #[error("An unspecified error occurred in the MCTS algorithm: {0}")]
    UnknownError(String),

    #[error("The provided index {0} is out of range. Valid range is 0 to {1}.")]
    OutOfRange(usize, usize),
}
