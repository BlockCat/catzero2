use mcts::GameState;

use crate::runner::RunnerError;

/// GamePlayed represents the result of a completed game, including the sequence of states,
/// policies, taken actions, final value, and the winner (if any).
#[derive(Clone, Debug)]
pub struct GamePlayed<G: GameState + Send + Sync> {
    /// Sequence of game states encountered during the game. S[i] corresponds to the state before taking action A[i].
    states: Vec<G>,
    /// Corresponding action policies for each state. The policy of state S[i] is stored in p[i].
    policies: Vec<Vec<(G::Action, f32)>>,
    /// Actions taken to transition between states. A[i] is the action taken from state S[i] to reach state S[i+1].
    taken_actions: Vec<G::Action>,
    /// Final value of the game from the perspective of the starting player.
    value: f32,
    /// The winner of the game, if any.
    result: GameResult,
}

#[derive(Clone, Debug)]
pub enum GameResult {
    Winner(i32),
    Draw,
}

unsafe impl<G: GameState + Send + Sync> Send for GamePlayed<G> {}
unsafe impl<G: GameState + Send + Sync> Sync for GamePlayed<G> {}

impl<G: GameState + Send + Sync> GamePlayed<G> {
    pub fn new(
        states: Vec<G>,
        policies: Vec<Vec<(G::Action, f32)>>,
        taken_actions: Vec<G::Action>,
        value: f32,
        result: GameResult,
    ) -> Result<Self, RunnerError> {
        if states.len() != policies.len() || states.len() != taken_actions.len() + 1 {
            return Err(RunnerError::GamePlayedError(format!(
                "Inconsistent lengths: states {}, policies {}, taken_actions {}",
                states.len(),
                policies.len(),
                taken_actions.len()
            )));
        }

        Ok(GamePlayed {
            states,
            policies,
            taken_actions,
            value,
            result,
        })
    }

    pub fn states(&self) -> &Vec<G> {
        &self.states
    }

    pub fn policies(&self) -> &Vec<Vec<(G::Action, f32)>> {
        &self.policies
    }

    pub fn taken_actions(&self) -> &Vec<G::Action> {
        &self.taken_actions
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn result(&self) -> GameResult {
        self.result.clone()
    }
}
