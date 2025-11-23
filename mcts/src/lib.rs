use rand::{rng, seq::IteratorRandom};
use std::{fmt::Debug, iter::zip, time::Duration};

pub use selection::{AlphaZeroSelectionStrategy, SelectionStrategy};
pub use tree::{DefaultAdjacencyTree, TreeHolder, TreeIndex};

pub mod selection;
mod tree;

/// Monte Carlo Tree Search implementation
/// Generic over:
/// - state S
/// - action A
/// - tree holder TH
/// - selection strategy SS
/// - state evaluation SE
#[derive(Clone)]
pub struct MCTS<S, A, TH, SS, SE>
where
    S: GameState<Action = A>,
    A: Clone,
    TH: TreeHolder<A>,
    SS: SelectionStrategy<S, A>,
    SE: StateEvaluation<S>,
{
    _phantom: std::marker::PhantomData<(S, A, TH)>,
    selection_strategy: SS,
    state_evaluation: SE,
    discount_factor: f64,
    tree: TH,
}

impl<
    S: GameState<Action = A>,
    A: Clone + Debug + PartialEq,
    TH: TreeHolder<A>,
    SS: SelectionStrategy<S, A>,
    SE: StateEvaluation<S>,
> MCTS<S, A, TH, SS, SE>
{
    pub fn new(selection_strategy: SS, state_evaluation: SE, discount_factor: f64) -> Self {
        MCTS {
            _phantom: std::marker::PhantomData,
            selection_strategy,
            state_evaluation,
            discount_factor,
            tree: TH::default(),
        }
    }

    pub async fn search_for_iterations_async(&mut self, state: &S, iterations: usize) -> Option<A> {
        let tree = &mut self.tree;

        if tree.is_empty() {
            tree.init_root_node();
        }

        for _ in 0..iterations {
            self.search_once(state).await;
        }

        self.best_from_tree()
    }

    pub fn search_for_iterations(&mut self, state: &S, iterations: usize) -> Option<A> {
        let mut tree = TH::default();
        tree.init_root_node();
        for _ in 0..iterations {
            futures::executor::block_on(self.search_once(state));
        }

        self.best_from_tree()
    }

    pub async fn search_for_duration_async(&mut self, state: &S, duration: Duration) -> Option<A> {
        let start = std::time::Instant::now();
        let mut tree = TH::default();
        tree.init_root_node();
        while start.elapsed() < duration {
            self.search_once(state).await;
        }

        self.best_from_tree()
    }

    pub fn search_for_duration(&mut self, state: &S, duration: Duration) -> Option<A> {
        let start = std::time::Instant::now();
        let mut tree = TH::default();
        tree.init_root_node();
        while start.elapsed() < duration {
            futures::executor::block_on(self.search_once(state));
        }

        self.best_from_tree()
    }

    fn best_from_tree(&self) -> Option<A> {
        let tree = &self.tree;
        let children = tree.children_visits(TreeIndex::root());
        let children_rewards = tree.children_rewards(TreeIndex::root());

        let average_rewards = children_rewards
            .iter()
            .zip(children.iter())
            .map(|(reward, visit)| {
                if *visit > 0 {
                    reward / *visit as f32
                } else {
                    f32::MIN
                }
            })
            .collect::<Vec<f32>>();

        let max_reward = average_rewards.iter().cloned().fold(f32::MIN, f32::max);

        let best_index: usize = average_rewards
            .iter()
            .enumerate()
            .filter(|(_, reward)| **reward == max_reward)
            .map(|(index, _)| index)
            .choose(&mut rng())?;

        let best_child = tree.child_index(TreeIndex::root(), best_index);

        Some(tree.action(best_child))
    }

    async fn search_once(&mut self, state: &S) {
        let (node, path, state, previous_state) = self.selection(state, &self.tree);

        if let Some(reward) = state.is_terminal() {
            self.backpropagation(path, reward);
            return;
        }

        let node_evaluation: ModelEvaluation = self
            .state_evaluation
            .evaluation(&state, &previous_state)
            .await;

        self.expansion(&state, node, node_evaluation.policy());
        self.backpropagation(path, node_evaluation.value());
    }

    fn selection(&self, state: &S, tree: &TH) -> (TreeIndex, Vec<TreeIndex>, S, Vec<S>) {
        let mut current_index = TreeIndex::root();
        let mut path = Vec::new();
        let mut state = state.clone();
        let mut previous_state = Vec::new();

        path.push(current_index);

        while tree.is_fully_expanded(current_index) {
            let (action, next_index) =
                self.selection_strategy
                    .select_child(tree, &state, current_index);

            current_index = next_index;
            path.push(current_index);

            previous_state.push(state.clone());
            state = state.take_action(action);
        }
        (current_index, path, state, previous_state)
    }

    fn expansion(&mut self, state: &S, node: TreeIndex, policy: &[f32]) {
        let possible_actions = state.get_possible_actions();

        if possible_actions.is_empty() {
            panic!("Node is in non terminal state, so actions are expected");
        }

        self.tree.expand_node(node, &possible_actions, policy);
    }

    fn backpropagation(&mut self, path: Vec<TreeIndex>, mut reward: f64) {
        for node_index in path.iter().rev() {
            self.tree.update_node_value(*node_index, reward);
            self.tree.increase_node_visit_count(*node_index);
            reward = -reward * self.discount_factor;
        }
    }

    pub fn subtree_pruning(&mut self, action: A) {
        let first_child_index = TreeIndex::new(1); // first child

        let action_index = self
            .tree
            .child_actions(TreeIndex::root())
            .iter()
            .position(|a| a.as_ref() == Some(&action))
            .expect("Action not found among root children");

        self.tree = self.tree.subtree(first_child_index.offset(action_index));
    }

    pub fn get_action_probabilities(&self) -> Vec<(A, f32)> {

        let actions = self.tree.child_actions(TreeIndex::root());
        let rewards = self.tree.children_rewards(TreeIndex::root());
        let visits = self.tree.children_visits(TreeIndex::root());

        zip(actions.iter(), zip(rewards.iter(), visits.iter()))
            .filter_map(|(action_opt, (reward, visit))| {
                if let Some(action) = action_opt {
                    Some((
                        action.clone(),
                        if *visit > 0 {
                            reward / *visit as f32
                        } else {
                            0.0
                        },
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<(A, f32)>>()
    }
}

pub trait GameState: Clone {
    type Action;

    fn get_possible_actions(&self) -> Vec<Self::Action>;
    fn take_action(&self, action: Self::Action) -> Self;

    /// return Some(reward) if terminal, None otherwise. +1 for win, -1 for loss, 0 for draw
    fn is_terminal(&self) -> Option<f64>;
}
pub trait StateEvaluation<S> {
    async fn evaluation(&self, state: &S, previous_state: &[S]) -> ModelEvaluation;
}

pub struct ModelEvaluation {
    policy: Vec<f32>,
    value: f64,
}

impl ModelEvaluation {
    pub fn new(policy: Vec<f32>, value: f64) -> Self {
        ModelEvaluation { policy, value }
    }
    pub fn value(&self) -> f64 {
        self.value
    }
    pub fn policy(&self) -> &Vec<f32> {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn expansion_test() {
    //     let mcts = create_mcts();
    //     let state = TestState::default();
    //     let possible_actions = state.get_possible_actions();
    //     assert_eq!(possible_actions.len(), 9);

    //     // mcts.expansion(&mut tree, &state, TreeIndex::root(), &vec![0.0; 9]);
    // }

    // fn create_mcts()
    // -> MCTS<TestState, TestAction, TestTreeHolder, TestSelectionStrategy, TestStateEvaluation> {
    //     MCTS::new(TestSelectionStrategy, TestStateEvaluation)
    // }
    pub struct TestSelectionStrategy;
    pub struct TestStateEvaluation;
    #[derive(Default)]
    pub struct TestTreeHolder;

    #[derive(Clone, Default)]
    pub struct TestState(Vec<TestAction>);

    #[derive(Clone, PartialEq, Eq, Hash)]
    pub struct TestAction(usize);

    impl GameState for TestState {
        type Action = TestAction;

        fn get_possible_actions(&self) -> Vec<Self::Action> {
            (0..20)
                .map(|i| TestAction(i))
                .filter(|a| !self.0.contains(a))
                .collect()
        }

        fn take_action(&self, _action: Self::Action) -> Self {
            self.clone()
        }

        fn is_terminal(&self) -> Option<f64> {
            None
        }
    }

    impl SelectionStrategy<TestState, TestAction> for TestSelectionStrategy {
        fn select_child(
            &self,
            _tree: &impl TreeHolder<TestAction>,
            _state: &TestState,
            _index: TreeIndex,
        ) -> (TestAction, TreeIndex) {
            (TestAction(0), TreeIndex::root())
        }
    }

    impl StateEvaluation<TestState> for TestStateEvaluation {
        async fn evaluation(
            &self,
            _state: &TestState,
            _previous_state: &[TestState],
        ) -> ModelEvaluation {
            ModelEvaluation::new(vec![0.0; 20], 0.0)
        }
    }
}
