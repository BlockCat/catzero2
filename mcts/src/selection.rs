use std::iter::zip;

use rand::{rng, seq::IteratorRandom};

use crate::tree::{TreeHolder, TreeIndex};

pub use alphazero::AlphaZeroSelectionStrategy;
pub use random::RandomSelectionStrategy;
pub use standard::StandardSelectionStrategy;

pub trait SelectionStrategy<S, A> {
    fn select_child(
        &self,
        tree: &impl TreeHolder<A>,
        state: &S,
        index: TreeIndex,
    ) -> (A, TreeIndex);
}

mod alphazero {
    use super::*;
    #[derive(Clone)]
    pub struct AlphaZeroSelectionStrategy {
        c1: f32,
        c2: f32,
        // _phantom: std::marker::PhantomData<(S, A)>,
    }

    impl AlphaZeroSelectionStrategy {
        pub fn new(c1: f32, c2: f32) -> Self {
            AlphaZeroSelectionStrategy {
                c1,
                c2,
                // _phantom: std::marker::PhantomData,
            }
        }
    }

    impl<S, A> SelectionStrategy<S, A> for AlphaZeroSelectionStrategy {
        fn select_child(
            &self,
            tree: &impl TreeHolder<A>,
            _state: &S,
            index: TreeIndex,
        ) -> (A, TreeIndex) {
            let parent_visit = tree.visit(index) as f32;
            let children_visits = tree.children_visits(index);
            let children_rewards = tree.children_rewards(index);
            let children_priors = tree.children_priors(index);
            let scores = alpha_zero_scores(
                parent_visit,
                children_visits,
                children_rewards,
                children_priors,
                self.c1,
                self.c2,
            );
            let max_score = *scores
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .expect("No children");

            let best_index = scores
                .iter()
                .enumerate()
                .filter(|(_, score)| **score == max_score)
                .map(|(index, _)| index)
                .choose(&mut rng())
                .expect("No best index found");

            let child_index = tree.child_index(index, best_index);
            let action = tree.action(child_index);

            (action, child_index)
        }
    }

    fn alpha_zero_scores(
        parent_visit: f32,
        children_visits: &[u32],
        children_rewards: &[f32],
        children_priors: &[f32],
        c1: f32,
        c2: f32,
    ) -> Vec<f32> {
        let exploration_term = f32::ln((1.0 + parent_visit + c1) / c1) + c2;
        let sqrt_parent = parent_visit.sqrt();

        let average_rewards = average_rewards(children_rewards, children_visits);
        let upper_bounds = calculate_upper_bounds(
            children_visits,
            children_priors,
            exploration_term,
            sqrt_parent,
        );

        zip(average_rewards, upper_bounds)
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>()
    }

    fn calculate_upper_bounds(
        children_visits: &[u32],
        children_priors: &[f32],
        exploration_term: f32,
        sqrt_parent: f32,
    ) -> impl Iterator<Item = f32> {
        children_priors
            .iter()
            .zip(children_visits.iter())
            .map(move |(prior, visit)| {
                exploration_term * *prior * sqrt_parent / (1.0 + *visit as f32)
            })
    }

    fn average_rewards(a: &[f32], b: &[u32]) -> impl Iterator<Item = f32> {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| if *y == 0 { 0.0 } else { x / (*y as f32) })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_average_rewards() {
            let rewards = vec![10.0, 20.0, 30.0];
            let visits = vec![2, 0, 5];
            let result: Vec<f32> = average_rewards(&rewards, &visits).collect();
            assert_eq!(result, vec![5.0, 0.0, 6.0]);
        }

        #[test]
        fn test_calculate_upper_bounds() {
            let children_visits = vec![1, 2, 3];
            let children_priors = vec![0.5, 0.3, 0.2];
            let exploration_term = 1.0;
            let sqrt_parent = 2.0;
            let result: Vec<f32> = calculate_upper_bounds(
                &children_visits,
                &children_priors,
                exploration_term,
                sqrt_parent,
            )
            .collect();
            assert_eq!(result, vec![0.5, 0.2, 0.1]);
        }

        #[test]
        fn test_alpha_zero_scores() {
            let parent_visit = 4.0;
            let children_visits = vec![2, 0, 5];
            let children_rewards = vec![0.7, 0.0, 1.0];
            let children_priors = vec![0.5, 0.3, 0.2];
            let c1 = 1.0;
            let c2 = 2.0;
            let result = alpha_zero_scores(
                parent_visit,
                &children_visits,
                &children_rewards,
                &children_priors,
                c1,
                c2,
            );
            assert_eq!(result.len(), 3);
            assert_eq!(result, vec![1.6139199, 2.275056, 0.452784]);
        }
    }
}

mod random {
    use super::*;
    pub struct RandomSelectionStrategy;

    impl<S, A> SelectionStrategy<S, A> for RandomSelectionStrategy {
        fn select_child(
            &self,
            tree: &impl TreeHolder<A>,
            _state: &S,
            index: TreeIndex,
        ) -> (A, TreeIndex) {
            let children_count = tree.children_visits(index).len();
            let chosen_index = (0..children_count)
                .choose(&mut rng())
                .expect("No children to choose from");
            let child_index = tree.child_index(index, chosen_index);
            let action = tree.action(child_index);
            (action, child_index)
        }
    }
}

mod standard {
    use super::*;
    pub struct StandardSelectionStrategy {
        pub exploration_constant: f32,
    }

    impl StandardSelectionStrategy {
        pub fn new(exploration_constant: f32) -> Self {
            StandardSelectionStrategy {
                exploration_constant,
            }
        }
    }

    impl<S, A> SelectionStrategy<S, A> for StandardSelectionStrategy {
        fn select_child(
            &self,
            tree: &impl TreeHolder<A>,
            _state: &S,
            index: TreeIndex,
        ) -> (A, TreeIndex) {
            let parent_visit = tree.visit(index) as f32;
            let children_visits = tree.children_visits(index);
            let children_rewards = tree.children_rewards(index);

            let scores = children_visits
                .iter()
                .zip(children_rewards.iter())
                .map(|(visit, reward)| {
                    let avg_reward = if *visit == 0 {
                        0.0
                    } else {
                        reward / (*visit as f32)
                    };
                    let exploration = self.exploration_constant
                        * (parent_visit.ln() / (1.0 + *visit as f32)).sqrt();
                    avg_reward + exploration
                })
                .collect::<Vec<_>>();

            let max_score = *scores
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .expect("No children");

            let best_index = scores
                .iter()
                .enumerate()
                .filter(|(_, score)| **score == max_score)
                .map(|(index, _)| index)
                .choose(&mut rng())
                .expect("No best index found");

            let child_index = tree.child_index(index, best_index);
            let action = tree.action(child_index);

            (action, child_index)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::*;
        use crate::tree::DefaultAdjacencyTree;

        #[derive(Clone)]
        struct MockState;

        #[test]
        fn test_standard_selection_strategy_creation() {
            let strategy = StandardSelectionStrategy::new(1.4);
            assert_eq!(strategy.exploration_constant, 1.4);
        }

        #[test]
        fn test_standard_selection_with_unvisited_child() {
            let mut tree: DefaultAdjacencyTree<char> = DefaultAdjacencyTree::default();
            tree.init_root_node().unwrap();

            let actions = vec!['a', 'b', 'c'];
            let policy = vec![0.3, 0.5, 0.2];
            tree.expand_node(TreeIndex::root(), &actions, &policy);

            // Visit the root
            tree.increase_node_visit_count(TreeIndex::root());

            let strategy = StandardSelectionStrategy::new(1.4);
            let state = MockState;
            let (action, child_idx) = strategy.select_child(&tree, &state, TreeIndex::root());

            // Should select one of the children
            assert!(action == 'a' || action == 'b' || action == 'c');
            assert!(child_idx.index() >= 1 && child_idx.index() <= 3);
        }

        #[test]
        fn test_standard_selection_prefers_best_reward() {
            let mut tree: DefaultAdjacencyTree<i32> = DefaultAdjacencyTree::default();
            tree.init_root_node().unwrap();

            let actions = vec![1, 2, 3];
            let policy = vec![0.3, 0.3, 0.4];
            tree.expand_node(TreeIndex::root(), &actions, &policy);

            // Set up visits and rewards
            tree.increase_node_visit_count(TreeIndex::root());
            tree.increase_node_visit_count(TreeIndex::root());
            tree.increase_node_visit_count(TreeIndex::root());

            let child1 = tree.child_index(TreeIndex::root(), 0);
            let child2 = tree.child_index(TreeIndex::root(), 1);
            let child3 = tree.child_index(TreeIndex::root(), 2);

            // Give child2 the best reward
            tree.increase_node_visit_count(child1);
            tree.update_node_value(child1, 0.3);

            tree.increase_node_visit_count(child2);
            tree.update_node_value(child2, 0.9);

            tree.increase_node_visit_count(child3);
            tree.update_node_value(child3, 0.2);

            // With very low exploration constant, should prefer best reward
            let strategy = StandardSelectionStrategy::new(0.01);
            let state = MockState;
            let (action, _) = strategy.select_child(&tree, &state, TreeIndex::root());

            assert_eq!(action, 2);
        }

        #[test]
        fn test_random_selection_strategy() {
            let mut tree: DefaultAdjacencyTree<char> = DefaultAdjacencyTree::default();
            tree.init_root_node().unwrap();

            let actions = vec!['x', 'y', 'z'];
            let policy = vec![0.3, 0.4, 0.3];
            tree.expand_node(TreeIndex::root(), &actions, &policy);

            let strategy = RandomSelectionStrategy;
            let state = MockState;

            // Test that it selects one of the valid children
            for _ in 0..10 {
                let (action, child_idx) = strategy.select_child(&tree, &state, TreeIndex::root());
                assert!(action == 'x' || action == 'y' || action == 'z');
                assert!(child_idx.index() >= 1 && child_idx.index() <= 3);
            }
        }
    }
}
