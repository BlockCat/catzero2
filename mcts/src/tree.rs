use std::collections::VecDeque;

use crate::error::{MCTSError::TreeNodeInitializationFailed, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TreeIndex(usize);

impl TreeIndex {
    pub fn root() -> Self {
        TreeIndex(0)
    }
    pub fn new(index: usize) -> Self {
        TreeIndex(index)
    }
    pub fn offset(&self, offset: usize) -> TreeIndex {
        TreeIndex(self.0 + offset)
    }
    pub fn index(&self) -> usize {
        self.0
    }
}

pub trait TreeHolder<A>: Default {
    /// Returns whether the tree is empty.
    fn is_empty(&self) -> bool;

    /// Returns the number of nodes in the tree.
    fn node_count(&self) -> usize;

    /// Initializes the root node of the tree.
    fn init_root_node(&mut self) -> Result<()>;
    /// Returns whether the node at `index` is fully expanded.
    /// Meaning, is there any child unvisited yet.
    fn is_fully_expanded(&self, index: TreeIndex) -> bool;

    /// Expands the node at `index` with the given possible actions, returning the index of the first
    /// child node.
    fn expand_node(&mut self, index: TreeIndex, actions: &[A], policy: &[f32]) -> TreeIndex;

    /// Updates the value of the node at `index` with the given reward.
    fn update_node_value(&mut self, index: TreeIndex, reward: f32);

    /// Increases the visit count of the node at `index` by one.
    fn increase_node_visit_count(&mut self, index: TreeIndex);

    fn child_index(&self, parent: TreeIndex, child_offset: usize) -> TreeIndex;

    fn action(&self, index: TreeIndex) -> A;

    /// Returns the visit count of the node at `index`.
    fn visit(&self, index: TreeIndex) -> u32;

    /// Returns the visit counts of the children of the node at `index`.
    fn children_visits(&self, index: TreeIndex) -> &[u32];
    /// Returns the rewards of the children of the node at `index`.
    fn children_rewards(&self, index: TreeIndex) -> &[f32];
    /// Returns the actions of the children of the node at `index`.
    fn child_actions(&self, index: TreeIndex) -> &[Option<A>];

    /// Returns the priors of the children of the node at `index`.
    fn children_priors(&self, index: TreeIndex) -> &[f32];

    fn subtree(&self, index: TreeIndex) -> Self;
}

#[derive(Clone)]
pub struct DefaultAdjacencyTree<A> {
    pub actions: Vec<Option<A>>,
    pub visit_counts: Vec<u32>,
    pub rewards: Vec<f32>,
    pub policy: Vec<f32>,

    pub children_start_index: Vec<Option<TreeIndex>>,
    pub children_count: Vec<u32>,
}

impl<A> Default for DefaultAdjacencyTree<A> {
    fn default() -> Self {
        DefaultAdjacencyTree {
            actions: Vec::new(),
            visit_counts: Vec::new(),
            rewards: Vec::new(),
            policy: Vec::new(),
            children_start_index: Vec::new(),
            children_count: Vec::new(),
        }
    }
}

impl<A: Clone> TreeHolder<A> for DefaultAdjacencyTree<A> {
    fn is_fully_expanded(&self, index: TreeIndex) -> bool {
        self.children_start_index[index.index()].is_some()
    }

    fn node_count(&self) -> usize {
        self.actions.len()
    }

    fn init_root_node(&mut self) -> Result<()> {
        if !self.actions.is_empty() {
            return Err(TreeNodeInitializationFailed);
        }
        self.actions.push(None);
        self.visit_counts.push(0);
        self.rewards.push(0.0);
        self.policy.push(0.0);
        self.children_start_index.push(None);
        self.children_count.push(0);

        Ok(())
    }

    fn expand_node(&mut self, index: TreeIndex, actions: &[A], policy: &[f32]) -> TreeIndex {
        debug_assert_eq!(
            actions.len(),
            policy.len(),
            "Actions and policy length must match"
        );
        let start_index = self.actions.len();
        self.children_start_index[index.index()] = Some(TreeIndex::new(start_index));
        self.children_count[index.index()] = actions.len() as u32;

        self.actions.extend(actions.iter().cloned().map(Some));
        self.visit_counts.extend(vec![0; actions.len()]);
        self.rewards.extend(vec![0.0; actions.len()]);
        self.policy.extend(policy.iter());
        self.children_start_index.extend(vec![None; actions.len()]);
        self.children_count.extend(vec![0; actions.len()]);

        TreeIndex::new(start_index)
    }

    fn update_node_value(&mut self, index: TreeIndex, reward: f32) {
        self.rewards[index.index()] += reward;
    }

    fn increase_node_visit_count(&mut self, index: TreeIndex) {
        self.visit_counts[index.index()] += 1;
    }

    fn child_index(&self, parent: TreeIndex, child_offset: usize) -> TreeIndex {
        self.children_start_index[parent.index()]
            .expect("Parent node is not expanded")
            .offset(child_offset)
    }

    fn action(&self, index: TreeIndex) -> A {
        self.actions[index.index()]
            .clone()
            .expect("Action not found")
    }

    fn visit(&self, index: TreeIndex) -> u32 {
        self.visit_counts[index.index()]
    }

    fn children_visits(&self, index: TreeIndex) -> &[u32] {
        let start_index = self.children_start_index[index.index()]
            .expect("Node is not expanded")
            .index();
        let count = self.children_count[index.index()] as usize;

        &self.visit_counts[start_index..start_index + count]
    }

    fn children_rewards(&self, index: TreeIndex) -> &[f32] {
        let start_index = self.children_start_index[index.index()]
            .expect("Node is not expanded")
            .index();
        let count = self.children_count[index.index()] as usize;

        &self.rewards[start_index..start_index + count]
    }

    fn children_priors(&self, index: TreeIndex) -> &[f32] {
        let start_index = self.children_start_index[index.index()]
            .expect("Node is not expanded")
            .index();
        let count = self.children_count[index.index()] as usize;

        &self.policy[start_index..start_index + count]
    }

    fn child_actions(&self, index: TreeIndex) -> &[Option<A>] {
        let start_index = self.children_start_index[index.index()]
            .expect("Node is not expanded")
            .index();
        let count = self.children_count[index.index()] as usize;

        &self.actions[start_index..start_index + count]
    }

    fn subtree(&self, index: TreeIndex) -> Self {
        let mut new_tree = DefaultAdjacencyTree::default();

        let mut position_queue: VecDeque<(Option<A>, TreeIndex)> = VecDeque::new();
        position_queue.push_back((self.actions[index.index()].clone(), index));

        while let Some((action, source_index)) = position_queue.pop_front() {
            let children_count = self.children_count[source_index.index()];

            new_tree.actions.push(action);
            new_tree.visit_counts.push(self.visit(source_index));
            new_tree.rewards.push(-self.rewards[source_index.index()]);
            new_tree.policy.push(self.policy[source_index.index()]);
            new_tree.children_count.push(children_count);

            let child_start =
                if let Some(child_start) = self.children_start_index[source_index.index()] {
                    let new_child_start = TreeIndex::new(new_tree.actions.len());
                    let offset = position_queue.len();

                    for child_source in
                        child_start.index()..(child_start.index() + children_count as usize)
                    {
                        let child_action = self.actions[child_source].clone();

                        position_queue.push_back((child_action, TreeIndex::new(child_source)));
                    }
                    Some(new_child_start.offset(offset))
                } else {
                    None
                };

            new_tree.children_start_index.push(child_start);
        }

        new_tree
    }

    fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tree_index() {
        let index = TreeIndex::new(5);
        assert_eq!(index.index(), 5);
    }

    #[test]
    fn test_tree_index_offset() {
        let index = TreeIndex::new(3);
        let offset_index = index.offset(2);
        assert_eq!(offset_index.index(), 5);
        assert_eq!(offset_index, TreeIndex::new(5));
    }

    #[test]
    fn test_default_adjacency_tree() {
        let mut tree: DefaultAdjacencyTree<char> = DefaultAdjacencyTree::default();
        tree.init_root_node().unwrap();
        assert_eq!(tree.visit(TreeIndex::root()), 0);
        assert!(!tree.is_fully_expanded(TreeIndex::root()));
    }

    #[test]
    fn test_expand_node() {
        let mut tree: DefaultAdjacencyTree<char> = DefaultAdjacencyTree::default();
        tree.init_root_node().unwrap();
        let actions = vec!['a', 'b', 'c'];
        let policy = vec![0.2, 0.5, 0.3];
        let child_index = tree.expand_node(TreeIndex::root(), &actions, &policy);
        assert!(tree.is_fully_expanded(TreeIndex::root()));
        assert_eq!(child_index.index(), 1);
    }

    #[test]
    fn test_update_and_visit() {
        let mut tree: DefaultAdjacencyTree<char> = DefaultAdjacencyTree::default();
        tree.init_root_node().unwrap();
        let actions = vec!['a', 'b'];
        let policy = vec![0.6, 0.4];
        let child_index = tree.expand_node(TreeIndex::root(), &actions, &policy);

        tree.increase_node_visit_count(child_index);
        tree.update_node_value(child_index, 1.0);

        assert_eq!(tree.visit(child_index), 1);
        assert_eq!(tree.rewards[child_index.index()], 1.0);
    }

    #[test]
    fn test_children_methods() {
        let mut tree: DefaultAdjacencyTree<char> = DefaultAdjacencyTree::default();
        tree.init_root_node().unwrap();
        let actions = vec!['x', 'y', 'z'];
        let policy = vec![0.3, 0.4, 0.3];
        tree.expand_node(TreeIndex::root(), &actions, &policy);

        for i in 0..3 {
            let child = tree.child_index(TreeIndex::root(), i);
            tree.increase_node_visit_count(child);
            tree.update_node_value(child, (i + 1) as f32);
        }

        let visits = tree.children_visits(TreeIndex::root());
        let rewards = tree.children_rewards(TreeIndex::root());
        let priors = tree.children_priors(TreeIndex::root());

        assert_eq!(visits, &[1, 1, 1]);
        assert_eq!(rewards, &[1.0, 2.0, 3.0]);
        assert_eq!(priors, &[0.3, 0.4, 0.3]);
    }

    #[test]
    fn test_subtree() {
        let mut tree: DefaultAdjacencyTree<char> = DefaultAdjacencyTree::default();

        // A -> B, C, D +
        // B -> E, F -
        // C -> G -
        // D -> [] -
        // E -> H, I, J +
        // F -> K, L +
        // G -> M +

        let nodes = vec![
            (Some('a'), 200, -200.0, 0.9, Some(TreeIndex::new(1)), 3),
            (Some('b'), 50, 50.0, 0.05, Some(TreeIndex::new(4)), 2),
            (Some('c'), 20, 20.0, 0.03, Some(TreeIndex::new(6)), 1),
            (Some('d'), 10, 10.0, 0.02, None, 0),
            (Some('e'), 5, -5.0, 0.5, Some(TreeIndex::new(7)), 3),
            (Some('f'), 6, -6.0, 0.5, Some(TreeIndex::new(10)), 2),
            (Some('g'), 7, -7.0, 0.6, None, 1),
            (Some('h'), 8, 8.0, 0.4, None, 0),
            (Some('i'), 9, 9.0, 0.7, None, 0),
            (Some('j'), 10, 10.0, 0.3, None, 0),
            (Some('k'), 11, 11.0, 0.8, None, 0),
            (Some('l'), 12, 12.0, 0.2, None, 0),
            (Some('m'), 13, 13.0, 1.0, None, 0),
        ];

        tree.actions
            .append(&mut nodes.iter().map(|x| x.0).collect());
        tree.visit_counts
            .append(&mut nodes.iter().map(|x| x.1).collect());
        tree.rewards
            .append(&mut nodes.iter().map(|x| x.2).collect());
        tree.policy.append(&mut nodes.iter().map(|x| x.3).collect());
        tree.children_start_index
            .append(&mut nodes.iter().map(|x| x.4).collect());
        tree.children_count
            .append(&mut nodes.iter().map(|x| x.5).collect());

        let subtree = tree.subtree(TreeIndex::new(1)); // Subtree rooted at B

        assert_eq!(subtree.actions.len(), 8);
        assert_eq!(subtree.visit_counts.len(), 8);
        assert_eq!(subtree.rewards.len(), 8);
        assert_eq!(subtree.policy.len(), 8);
        assert_eq!(subtree.children_start_index.len(), 8);
        assert_eq!(subtree.children_count.len(), 8);

        assert_eq!(
            subtree.actions,
            vec![
                Some('b'),
                Some('e'),
                Some('f'),
                Some('h'),
                Some('i'),
                Some('j'),
                Some('k'),
                Some('l'),
            ]
        );
        assert_eq!(subtree.visit_counts, vec![50, 5, 6, 8, 9, 10, 11, 12]);
        assert_eq!(
            subtree.rewards,
            vec![-50.0, 5.0, 6.0, -8.0, -9.0, -10.0, -11.0, -12.0]
        );
        assert_eq!(
            subtree.policy,
            vec![0.05, 0.5, 0.5, 0.4, 0.7, 0.3, 0.8, 0.2]
        );

        // B -> E, F -
        // E -> H, I, J +
        // F -> K, L +

        // B,E,F,H,I,J,K,L
        assert_eq!(
            subtree.children_start_index,
            vec![
                Some(TreeIndex::new(1)),
                Some(TreeIndex::new(3)),
                Some(TreeIndex::new(6)),
                None,
                None,
                None,
                None,
                None
            ]
        );
        assert_eq!(subtree.children_count, vec![2, 3, 2, 0, 0, 0, 0, 0]);
    }
}
