use mcts::GameState;

#[derive(Clone, Debug)]
pub struct GamePlayed<G: GameState + Send + Sync> {
    states: Vec<G>,
    policies: Vec<Vec<(G::Action, f32)>>,
    taken_actions: Vec<G::Action>,
    value: f32,
    winner: Option<i32>,
}

unsafe impl<G: GameState + Send + Sync> Send for GamePlayed<G> {}
unsafe impl<G: GameState + Send + Sync> Sync for GamePlayed<G> {}

impl<G: GameState + Send + Sync> GamePlayed<G> {
    pub fn new(
        states: Vec<G>,
        policies: Vec<Vec<(G::Action, f32)>>,
        taken_actions: Vec<G::Action>,
        value: f32,
        winner: Option<i32>,
    ) -> Self {
        GamePlayed {
            states,
            policies,
            taken_actions,
            value,
            winner,
        }
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

    pub fn winner(&self) -> Option<i32> {
        self.winner
    }
}
