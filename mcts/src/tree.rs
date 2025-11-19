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

    /// Initializes the root node of the tree.
    fn init_root_node(&mut self);
    /// Returns whether the node at `index` is fully expanded.
    /// Meaning, is there any child unvisited yet.
    fn is_fully_expanded(&self, index: TreeIndex) -> bool;

    /// Expands the node at `index` with the given possible actions, returning the index of the first
    /// child node.
    fn expand_node(&mut self, index: TreeIndex, actions: &[A], policy: &[f32]) -> TreeIndex;

    /// Updates the value of the node at `index` with the given reward.
    fn update_node_value(&mut self, index: TreeIndex, reward: f64);

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

    /// Returns the priors of the children of the node at `index`.
    fn children_priors(&self, index: TreeIndex) -> &[f32];
}

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

    fn init_root_node(&mut self) {
        debug_assert!(self.actions.is_empty(), "Tree already initialized");
        self.actions.push(None);
        self.visit_counts.push(0);
        self.rewards.push(0.0);
        self.policy.push(0.0);
        self.children_start_index.push(None);
        self.children_count.push(0);    
    }

    fn expand_node(&mut self, index: TreeIndex, actions: &[A], policy: &[f32]) -> TreeIndex {
        debug_assert_eq!(actions.len(), policy.len(), "Actions and policy length must match");
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

    fn update_node_value(&mut self, index: TreeIndex, reward: f64) {
        self.rewards[index.index()] += reward as f32;
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
        self.actions[index.index()].clone().expect("Action not found")
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
}
