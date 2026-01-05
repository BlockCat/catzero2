use crate::error::MCTSError::{self, MoveNotFound};
use rand::{rng, seq::IteratorRandom};
use std::{collections::HashMap, fmt::Debug, future::Future, iter::zip, time::Duration};

pub use error::{MCTSError as Error, Result};
pub use selection::{AlphaZeroSelectionStrategy, SelectionStrategy};
pub use tree::{DefaultAdjacencyTree, TreeHolder, TreeIndex};

pub mod error;
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
    discount_factor: f32,
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
    pub fn new(selection_strategy: SS, state_evaluation: SE, discount_factor: f32) -> Result<Self> {
        let mut tree = TH::default();
        tree.init_root_node()?;
        Ok(MCTS {
            _phantom: std::marker::PhantomData,
            selection_strategy,
            state_evaluation,
            discount_factor,
            tree,
        })
    }

    pub fn positions_expanded(&self) -> usize {
        self.tree.node_count()
    }

    pub async fn search_for_iterations_async(&mut self, state: &S, iterations: usize) -> Result<A> {
        let tree = &mut self.tree;

        let positions_left = iterations.saturating_sub(tree.node_count()) + 1;
        for _ in 0..positions_left {
            self.search_once(state).await?;
        }

        self.best_from_tree()
    }

    pub fn search_for_iterations(&mut self, state: &S, iterations: usize) -> Result<A> {
        let tree = &mut self.tree;

        let positions_left = iterations.saturating_sub(tree.node_count()) + 1;
        for _ in 0..positions_left {
            futures::executor::block_on(self.search_once(state))?;
        }

        self.best_from_tree()
    }

    pub async fn search_for_duration_async(&mut self, state: &S, duration: Duration) -> Result<A> {
        let start = std::time::Instant::now();

        while start.elapsed() < duration {
            self.search_once(state).await?;
        }

        self.best_from_tree()
    }

    pub fn search_for_duration(&mut self, state: &S, duration: Duration) -> Result<A> {
        let start = std::time::Instant::now();

        while start.elapsed() < duration {
            futures::executor::block_on(self.search_once(state))?;
        }

        self.best_from_tree()
    }

    fn best_from_tree(&self) -> Result<A> {
        let tree = &self.tree;
        let children = tree.children_visits(TreeIndex::root());
        let children_rewards = tree.children_rewards(TreeIndex::root());

        debug_assert_eq!(
            children.len(),
            children_rewards.len(),
            "Children visits and rewards length mismatch"
        );

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
            .choose(&mut rng())
            .ok_or(MoveNotFound)?;

        let best_child = tree.child_index(TreeIndex::root(), best_index);

        Ok(tree.action(best_child))
    }

    /// Perform a single MCTS search iteration, returning the index of the expanded node
    async fn search_once(&mut self, state: &S) -> Result<TreeIndex> {
        let (node, path, state, previous_state) = self.selection(state, &self.tree);

        if let Some(reward) = state.is_terminal() {
            self.backpropagation(path, reward);
            return Ok(node);
        }

        let node_evaluation: ModelEvaluation<S::Action> = self
            .state_evaluation
            .evaluation(&state, &previous_state)
            .await?;

        self.expansion(node, node_evaluation.policy())?;
        self.backpropagation(path, node_evaluation.value());

        Ok(node)
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

    fn expansion(&mut self, node: TreeIndex, policy: &HashMap<A, f32>) -> Result<()> {
        if policy.is_empty() {
            tracing::error!("Expansion policy is empty");
            return Err(MCTSError::ExpansionError);
        }

        let (possible_actions, policy): (Vec<A>, Vec<f32>) =
            policy.iter().map(|(a, b)| (a.clone(), *b)).unzip();

        self.tree.expand_node(node, &possible_actions, &policy);
        Ok(())
    }

    fn backpropagation(&mut self, path: Vec<TreeIndex>, mut reward: f32) {
        for node_index in path.iter().rev() {
            self.tree.update_node_value(*node_index, reward);
            self.tree.increase_node_visit_count(*node_index);
            reward = -reward * self.discount_factor;
        }
    }

    pub fn subtree_pruning(&mut self, action: A) -> Result<()> {
        let first_child_index = TreeIndex::new(1); // first child

        let action_index = self
            .tree
            .child_actions(TreeIndex::root())
            .iter()
            .position(|a| a.as_ref() == Some(&action))
            .ok_or(MCTSError::MoveNotFound)?;

        self.tree = self.tree.subtree(first_child_index.offset(action_index));

        Ok(())
    }

    pub fn get_action_probabilities(&self) -> Vec<(A, f32)> {
        let actions = self.tree.child_actions(TreeIndex::root());
        let rewards = self.tree.children_rewards(TreeIndex::root());
        let visits = self.tree.children_visits(TreeIndex::root());

        zip(actions.iter(), zip(rewards.iter(), visits.iter()))
            .filter_map(|(action_opt, (reward, visit))| {
                action_opt.as_ref().map(|action| {
                    (
                        action.clone(),
                        if *visit > 0 {
                            reward / *visit as f32
                        } else {
                            0.0
                        },
                    )
                })
            })
            .collect::<Vec<(A, f32)>>()
    }
}

pub trait GameState: Clone + Default {
    type Action;

    fn current_player_id(&self) -> usize;

    fn get_possible_actions(&self) -> Vec<Self::Action>;
    fn take_action(&self, action: Self::Action) -> Self;

    /// return Some(reward) if terminal, None otherwise. +1 for win, -1 for loss, 0 for draw
    fn is_terminal(&self) -> Option<f32>;
}
pub trait StateEvaluation<S: GameState> {
    fn evaluation(
        &self,
        state: &S,
        previous_state: &[S],
    ) -> impl Future<Output = Result<ModelEvaluation<S::Action>>> + Send;
}

pub struct ModelEvaluation<C> {
    policy: HashMap<C, f32>,
    value: f32,
}

impl<C> ModelEvaluation<C> {
    pub fn new(policy: HashMap<C, f32>, value: f32) -> Self {
        ModelEvaluation { policy, value }
    }
    pub fn value(&self) -> f32 {
        self.value
    }
    pub fn policy(&self) -> &HashMap<C, f32> {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selection::StandardSelectionStrategy;

    // Simple test game state for testing MCTS
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestGame {
        value: i32,
        depth: usize,
        max_depth: usize,
    }

    impl Default for TestGame {
        fn default() -> Self {
            TestGame {
                value: 0,
                depth: 0,
                max_depth: 3,
            }
        }
    }

    impl GameState for TestGame {
        type Action = i32;

        fn get_possible_actions(&self) -> Vec<Self::Action> {
            if self.depth >= self.max_depth {
                vec![]
            } else {
                vec![1, 2, 3]
            }
        }

        fn take_action(&self, action: Self::Action) -> Self {
            TestGame {
                value: self.value + action,
                depth: self.depth + 1,
                max_depth: self.max_depth,
            }
        }

        fn is_terminal(&self) -> Option<f32> {
            if self.depth >= self.max_depth {
                Some(self.value as f32 / 10.0)
            } else {
                None
            }
        }

        fn current_player_id(&self) -> usize {
            self.depth % 2
        }
    }

    struct TestEvaluator;

    impl StateEvaluation<TestGame> for TestEvaluator {
        async fn evaluation(
            &self,
            state: &TestGame,
            _previous_state: &[TestGame],
        ) -> Result<ModelEvaluation<i32>> {
            let actions = state.get_possible_actions();
            let mut policy = HashMap::new();

            for action in actions {
                policy.insert(action, 1.0 / 3.0);
            }

            Ok(ModelEvaluation::new(policy, 0.5))
        }
    }

    #[test]
    fn test_mcts_initialization() {
        let selection_strategy = StandardSelectionStrategy::new(1.4);
        let evaluator = TestEvaluator;
        let mcts: MCTS<TestGame, i32, DefaultAdjacencyTree<i32>, _, _> =
            MCTS::new(selection_strategy, evaluator, 0.9).unwrap();

        assert_eq!(mcts.positions_expanded(), 1); // Root node
    }

    #[test]
    fn test_mcts_search_iterations() {
        let selection_strategy = StandardSelectionStrategy::new(1.4);
        let evaluator = TestEvaluator;
        let mut mcts: MCTS<TestGame, i32, DefaultAdjacencyTree<i32>, _, _> =
            MCTS::new(selection_strategy, evaluator, 0.9).unwrap();

        let game = TestGame::default();
        let action = mcts.search_for_iterations(&game, 10);

        assert!(action.is_ok());
        assert!(mcts.positions_expanded() >= 10);
    }

    #[test]
    fn test_mcts_search_duration() {
        let selection_strategy = StandardSelectionStrategy::new(1.4);
        let evaluator = TestEvaluator;
        let mut mcts: MCTS<TestGame, i32, DefaultAdjacencyTree<i32>, _, _> =
            MCTS::new(selection_strategy, evaluator, 0.9).unwrap();

        let game = TestGame::default();
        let duration = Duration::from_millis(10);
        let action = mcts.search_for_duration(&game, duration);

        assert!(action.is_ok());
        assert!(mcts.positions_expanded() > 1);
    }

    #[test]
    fn test_mcts_best_from_tree() {
        let selection_strategy = StandardSelectionStrategy::new(1.4);
        let evaluator = TestEvaluator;
        let mut mcts: MCTS<TestGame, i32, DefaultAdjacencyTree<i32>, _, _> =
            MCTS::new(selection_strategy, evaluator, 0.9).unwrap();

        let game = TestGame::default();
        mcts.search_for_iterations(&game, 50).unwrap();

        let action = mcts.best_from_tree();
        assert!(action.is_ok());
        let action = action.unwrap();
        assert!(action == 1 || action == 2 || action == 3);
    }

    #[test]
    fn test_mcts_get_action_probabilities() {
        let selection_strategy = StandardSelectionStrategy::new(1.4);
        let evaluator = TestEvaluator;
        let mut mcts: MCTS<TestGame, i32, DefaultAdjacencyTree<i32>, _, _> =
            MCTS::new(selection_strategy, evaluator, 0.9).unwrap();

        let game = TestGame::default();
        mcts.search_for_iterations(&game, 50).unwrap();

        let probs = mcts.get_action_probabilities();
        assert_eq!(probs.len(), 3);

        // All actions should be present
        let actions: Vec<i32> = probs.iter().map(|(a, _)| *a).collect();
        assert!(actions.contains(&1));
        assert!(actions.contains(&2));
        assert!(actions.contains(&3));
    }

    #[test]
    fn test_mcts_subtree_pruning() {
        let selection_strategy = StandardSelectionStrategy::new(1.4);
        let evaluator = TestEvaluator;
        let mut mcts: MCTS<TestGame, i32, DefaultAdjacencyTree<i32>, _, _> =
            MCTS::new(selection_strategy, evaluator, 0.9).unwrap();

        let game = TestGame::default();
        mcts.search_for_iterations(&game, 50).unwrap();

        let initial_size = mcts.positions_expanded();

        let action = mcts.best_from_tree().unwrap();
        mcts.subtree_pruning(action).unwrap();

        let after_pruning_size = mcts.positions_expanded();
        assert!(after_pruning_size < initial_size);
    }

    #[test]
    fn test_model_evaluation_creation() {
        let mut policy = HashMap::new();
        policy.insert(1, 0.5);
        policy.insert(2, 0.3);
        policy.insert(3, 0.2);

        let evaluation = ModelEvaluation::new(policy.clone(), 0.8);

        assert_eq!(evaluation.value(), 0.8);
        assert_eq!(evaluation.policy(), &policy);
    }

    #[test]
    fn test_mcts_terminal_state_immediate() {
        // Create a terminal game state
        let game = TestGame {
            value: 10,
            depth: 3,
            max_depth: 3,
        };

        // For terminal state with no possible actions
        assert_eq!(game.get_possible_actions().len(), 0);
        assert!(game.is_terminal().is_some());
    }

    #[test]
    fn test_backpropagation_discount() {
        let selection_strategy = StandardSelectionStrategy::new(1.4);
        let evaluator = TestEvaluator;
        let mut mcts: MCTS<TestGame, i32, DefaultAdjacencyTree<i32>, _, _> =
            MCTS::new(selection_strategy, evaluator, 0.5).unwrap(); // discount factor 0.5

        let game = TestGame::default();
        mcts.search_for_iterations(&game, 10).unwrap();

        // Just verify search completes with discount factor
        assert!(mcts.positions_expanded() >= 10);
    }
}
