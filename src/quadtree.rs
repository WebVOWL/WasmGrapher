use glam::Vec2;
use rand::prelude::*;
use smallvec::{SmallVec, smallvec};
use std::{
    collections::HashMap,
    fmt::format,
    mem::{swap, take},
    ops::Range,
};
const EPSILON: f32 = 1e-3;
const UNINITIALIZED: u32 = u32::MAX;

/// The dimension of the quadtree
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoundingBox2D {
    pub center: Vec2,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox2D {
    #[must_use]
    pub const fn new(center: Vec2, width: f32, height: f32) -> Self {
        Self {
            center,
            width,
            height,
        }
    }

    /// Returns the index of `loc` into the indices of a [`Node::Root`].
    #[must_use]
    pub fn section(&self, loc: Vec2) -> u8 {
        let mut section = 0x00;

        if loc[1] > self.center[1] {
            section |= 0b10;
        }

        if loc[0] > self.center[0] {
            section |= 0b01;
        }

        section
    }

    /// Creates a sub-quadrant from `section`, which is computed by [`Self::section`].
    #[must_use]
    pub fn sub_quadrant(&self, section: u8) -> Self {
        let mut shift = self.center;
        if section & 0b01 > 0 {
            shift[0] += 0.25 * self.width;
        } else {
            shift[0] -= 0.25 * self.width;
        }

        if section & 0b10 > 0 {
            shift[1] += 0.25 * self.height;
        } else {
            shift[1] -= 0.25 * self.height;
        }
        Self {
            center: shift,
            width: self.width * 0.5,
            height: self.height * 0.5,
        }
    }
}

#[derive(Debug)]
pub enum Node {
    Root {
        indices: [u32; 4],
        mass: f32,
        pos: Vec2,
    },
    Leaf {
        mass: f32,
        pos: Vec2,
        id: u32, // Not Option<u32> to save 8 bytes on each Leaf
    },
}

impl Node {
    #[must_use]
    const fn new_leaf(pos: Vec2, mass: f32, id: u32) -> Self {
        Self::Leaf { mass, pos, id }
    }
    #[must_use]
    const fn new_root(pos: Vec2, mass: f32, indices: [u32; 4]) -> Self {
        Self::Root { indices, mass, pos }
    }

    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    #[must_use]
    pub const fn is_root(&self) -> bool {
        matches!(self, Self::Root { .. })
    }

    #[must_use]
    pub fn position(&self) -> Vec2 {
        match self {
            Self::Root { pos, mass, .. } => pos / mass,
            Self::Leaf { pos, .. } => *pos,
        }
    }

    #[must_use]
    pub const fn mass(&self) -> f32 {
        match self {
            Self::Root { mass, .. } | Self::Leaf { mass, .. } => *mass,
        }
    }

    #[must_use]
    pub const fn id(&self) -> Option<u32> {
        match self {
            Self::Root { .. } => None,
            Self::Leaf { id, .. } => Some(*id),
        }
    }
}

#[derive(Debug, Default)]
/// A quadtree capable of storing [`u32::MAX`] elements.
pub struct QuadTree {
    children: HashMap<u32, Node>,
    pub boundary: BoundingBox2D,
    root: u32,
}

impl QuadTree {
    #[must_use]
    pub fn new(boundary: BoundingBox2D) -> Self {
        Self {
            root: 0,
            boundary,
            children: HashMap::new(),
        }
    }

    #[must_use]
    pub fn with_capacity(boundary: BoundingBox2D, capacity: usize) -> Self {
        Self {
            root: 0,
            boundary,
            children: HashMap::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, new_pos: Vec2, new_mass: f32) -> Result<(), String> {
        self.insert_id(self.children.len() as u32, new_pos, new_mass)
    }

    /// Inserts a new point with a unique ID into the quadtree.
    pub fn insert_id(&mut self, uid: u32, new_pos: Vec2, new_mass: f32) -> Result<(), String> {
        if self.children.len() == (UNINITIALIZED - 1) as usize {
            return Err(format!(
                "Quadtree is full! (storing {}/{} elements)",
                self.children.len(),
                UNINITIALIZED - 1
            ));
        }

        // With only one node there's no need to continue
        if self.children.is_empty() {
            let new_leaf = Node::new_leaf(new_pos, new_mass, uid);
            self.children.insert(self.children.len() as u32, new_leaf);
            return Ok(());
        }

        let mut bb = self.boundary.clone();
        let mut root_index = self.root;
        let new_index = self.children.len() as u32;

        // Traversing the tree until we find an empty quadrant for the new leaf.
        while let Node::Root {
            indices, mass, pos, ..
        } = &mut self
            .children
            .get_mut(&root_index)
            .ok_or_else(|| format!("Failed to get index {root_index} while inserting"))?
        {
            // Update Mass and Pos of root to account for the new leaf.
            *mass += new_mass;
            *pos += new_pos * new_mass;

            let section = bb.section(new_pos);
            // If section not set: create new leaf and exit
            if indices[section as usize] == UNINITIALIZED {
                indices[section as usize] = new_index;
                break;
            }

            root_index = indices[section as usize];
            bb = bb.sub_quadrant(section);
        }

        // If new leaf is too close to current leaf we merge
        if let Node::Leaf { mass, pos, id } = self.children[&root_index]
            && pos.distance_squared(new_pos) < EPSILON
        {
            let m: f32 = mass + new_mass;
            self.children
                .entry(root_index)
                .and_modify(|node| *node = Node::new_leaf(pos, m, id));
            return Ok(());
        }

        self.children.insert(
            self.children.len() as u32,
            Node::new_leaf(new_pos, new_mass, uid),
        );

        // Create new root until leaf and new leaf are in different sections
        while let Node::Leaf { mass, pos, id } = self.children[&root_index] {
            let mut fin = false;

            // Pushes the old leaf to the back of the vector and inserts its index into the index array of the new root
            let old_node = Node::new_leaf(pos, mass, id);
            let old_index = self.children.len() as u32;
            self.children.insert(old_index, old_node);

            let section = bb.section(pos);
            let mut ind = [UNINITIALIZED, UNINITIALIZED, UNINITIALIZED, UNINITIALIZED];
            ind[section as usize] = old_index;

            let section = bb.section(new_pos);

            // If section of the new root is empty we can set it and exit
            if ind[section as usize] == UNINITIALIZED {
                ind[section as usize] = new_index;
                fin = true;
            }

            // Sets the old leaf index to the new root (thus the leaf index of the old leaf's parent now points to the new root)
            let new_root = Node::new_root(pos * mass + new_pos * new_mass, mass + new_mass, ind);
            self.children.insert(root_index, new_root);

            if fin {
                break;
            }

            root_index = old_index;

            bb = bb.sub_quadrant(section);
        }

        Ok(())
    }

    /// Removes a node and all its descendants from the quadtree.
    fn delete_index(&mut self, index: u32) -> Result<(), String> {
        let mut stack: SmallVec<[u32; 32]> = SmallVec::with_capacity(32);
        stack.push(index);
        let mut new_stack: SmallVec<[u32; 32]> = SmallVec::with_capacity(32);
        'outer: loop {
            for node_index in &stack {
                let node = self.children.remove(node_index).ok_or_else(|| {
                    format!("Failed to delete node at index {node_index}: index not found")
                })?;
                match node {
                    Node::Root { indices, .. } => {
                        for i in indices {
                            if i != UNINITIALIZED {
                                new_stack.push(i);
                            }
                        }
                    }
                    // A leaf has no descendants. Nothing to do.
                    Node::Leaf { .. } => {}
                }
            }
            if new_stack.is_empty() {
                break 'outer;
            }
            // Clearing and swapping values to keep memory allocations.
            stack.clear();
            swap(&mut stack, &mut new_stack);
        }
        Ok(())
    }

    /// Unassigns the index of a leaf node in its parent, making it the uninitialized value.
    ///
    /// If an update leaves all children of a root node uninitialized, the root node is deleted.
    /// This proceeds recursively up towards the root of the quadtree.
    ///
    /// Note that unassigned leaf nodes are NOT deleted.
    fn unassign_index(
        &mut self,
        section: u8,
        root_index: u32,
        parents: &[u32],
    ) -> Result<(), String> {
        if let Some(Node::Root { indices, pos, mass }) = &mut self.children.get_mut(&root_index) {
            indices[section as usize] = UNINITIALIZED;

            if indices.iter().all(|i| *i == UNINITIALIZED) {
                // All children are uninitialized. We can delete the node.
                self.delete_index(root_index);

                let last = parents.len().saturating_sub(1);
                let before_last = last.saturating_sub(1);

                if let (Some(new_root_index), Some(new_parents)) =
                    (parents.get(last), parents.get(0..before_last))
                {
                    // Unassign the deleted node from its parent.
                    self.unassign_index(section, *new_root_index, new_parents)?;
                }
            }
            return Ok(());
        }

        if let Some(Node::Leaf { .. }) = &self.children.get(&root_index) {
            let last = parents.len().saturating_sub(1);
            let before_last = last.saturating_sub(1);

            if let (Some(new_root_index), Some(new_parents)) =
                (parents.get(last), parents.get(0..before_last))
            {
                // Unassign the deleted node from its parent.
                self.unassign_index(section, *new_root_index, new_parents)?;
            }

            return Ok(());
        }
        Err(format!(
            "Cannot unassign index {root_index}: index not found"
        ))
    }

    /// Removes the point `delete_pos` from the quadtree, if it exists.
    pub fn delete_point(&mut self, delete_pos: Vec2) -> Result<(), String> {
        let Some(delete_node) = self.query_point(delete_pos) else {
            return Err(format!(
                "Failed to delete point '{delete_pos}': point not found"
            ));
        };

        let mut parent_indices: SmallVec<[u32; 128]> = smallvec![];

        let mut bb = self.boundary.clone();
        let mut section = bb.section(delete_pos);
        let mut current_index = self.root;
        let del_mass = delete_node.mass();

        // Traversing the tree until we find `delete_pos`.
        while let Node::Root { indices, mass, pos } =
            &mut self.children.get_mut(&current_index).ok_or_else(|| {
                format!("Failed to get index {current_index} while deleting point {delete_pos}")
            })?
        {
            // Update Mass and Pos of root to account for the removed node (when we find it).
            *mass -= del_mass;
            *pos -= delete_pos * del_mass;

            parent_indices.push(current_index);

            section = bb.section(delete_pos);
            current_index = indices[section as usize];
            bb = bb.sub_quadrant(section);
        }

        // The leaf node has a parent.
        if !parent_indices.is_empty() {
            // Set the deleted node to uninitialized in the parent node.
            self.unassign_index(section, current_index, parent_indices.as_slice())?;
        }

        self.delete_index(current_index)?;

        Ok(())
    }

    /// Returns the quadtree node having position `query_pos`.
    #[must_use]
    pub fn query_point(&self, query_pos: Vec2) -> Option<&Node> {
        let mut bb = self.boundary.clone();
        let mut root_index = self.root;

        // Traversing the tree until we find `query_pos`.
        while let Some(Node::Root { indices, pos, .. }) = &self.children.get(&root_index) {
            let section = bb.section(query_pos);
            root_index = indices[section as usize];

            if root_index == UNINITIALIZED {
                return None;
            }

            bb = bb.sub_quadrant(section);
        }

        // Check the final leaf node
        if let Some(Node::Leaf { pos, .. }) = &self.children.get(&root_index)
            && pos.distance_squared(query_pos) <= EPSILON
        {
            return Some(&self.children[&root_index]);
        }
        None
    }

    /// Returns true if `contain_pos` is a point in the quadtree.
    #[must_use]
    pub fn contains(&self, contain_pos: Vec2) -> bool {
        self.query_point(contain_pos).is_some()
    }

    /// Clears the quadtree, removing all elements.
    ///
    /// Note that this method has no effect on the allocated capacity of the quadtree.
    pub fn clear(&mut self) {
        self.root = 0;
        self.children.clear();
    }

    /// Returns the amount of leaves in the tree.
    ///
    /// This the "visible" size of tree and corresponds to the number of points inserted.
    ///
    /// The total size of the tree is [`Self::roots`] + [`Self::leaves`].
    /// However, please use [`Self::len`] for this, as it is much faster.
    ///
    /// This method's runtime is O(n).
    #[must_use]
    pub fn leaves(&self) -> usize {
        let mut count = 0;
        for (index, node) in &self.children {
            if let Node::Leaf { .. } = node {
                count += 1;
            }
        }
        count
    }

    /// Returns the amount of root nodes in the tree.
    ///
    /// This the "hidden" size of tree, as root nodes are not inserted, but generated automatically.
    ///
    /// The total size of the tree is [`Self::roots`] + [`Self::leaves`].
    /// However, please use [`Self::len`] for this, as it is much faster.
    ///
    /// This method's runtime is O(n).
    #[must_use]
    pub fn roots(&self) -> usize {
        let mut count = 0;
        for (index, node) in &self.children {
            if let Node::Root { .. } = node {
                count += 1;
            }
        }
        count
    }

    /// Returns the number of nodes in the tree.
    ///
    /// This method's runtime is O(1).
    #[must_use]
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Returns `true` if the tree contains no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    /// Returns the net force acted upon the body for the entire tree.
    /// This is calculated using the Barnes-Hut algorithm.
    /// `uid` is the unique ID of the body to compute forces on.
    ///
    /// `position` is position of the body.
    ///
    /// `mass` is the mass of the body.
    ///
    /// `theta` is the threshold of the Barnes-Hut algorithm.
    ///
    /// `repel_force` is the magnitude of the repulsive force between two bodies (the Coulomb constant).
    #[must_use]
    pub fn approximate_forces_on_body(
        &self,
        uid: u32,
        position: Vec2,
        mass: f32,
        theta: f32,
        repel_force: f32,
    ) -> Vec2 {
        let mut net_force = Vec2::ZERO;

        if self.children.is_empty() {
            return net_force;
        }

        let mut s: f32 = self.boundary.width.max(self.boundary.height);

        let mut stack: SmallVec<[u32; 32]> = SmallVec::new();
        stack.push(self.root);
        let mut new_stack: SmallVec<[u32; 32]> = SmallVec::with_capacity(32);
        'outer: loop {
            for node_index in &stack {
                let parent = &self.children[node_index];

                match parent {
                    Node::Root { indices, .. } => {
                        let center_mass = parent.position();
                        let dist = center_mass.distance(position);
                        if s / dist < theta {
                            net_force += Self::repel_force(
                                position,
                                center_mass,
                                mass,
                                parent.mass(),
                                repel_force,
                            );
                        } else {
                            for i in indices {
                                if *i != UNINITIALIZED {
                                    new_stack.push(*i);
                                }
                            }
                        }
                    }
                    Node::Leaf { id, .. } => {
                        if uid != *id {
                            net_force += Self::repel_force(
                                position,
                                parent.position(),
                                mass,
                                parent.mass(),
                                repel_force,
                            );
                        }
                    }
                }
            }
            if new_stack.is_empty() {
                break 'outer;
            }
            s *= 0.5;

            // Clearing and swapping values to keep memory allocations.
            stack.clear();
            swap(&mut stack, &mut new_stack);
        }
        net_force.clamp(
            Vec2::new(-100_000.0, -100_000.0),
            Vec2::new(100_000.0, 100_000.0),
        )
    }

    /// Computes the electrostatic force between two bodies. Based on Coulomb's law.
    fn repel_force(pos1: Vec2, pos2: Vec2, mass1: f32, mass2: f32, repel_force: f32) -> Vec2 {
        let mut component_form = pos2 - pos1;
        let mut length_sqr = component_form.length_squared();

        // Limit forces for very close points; randomize direction if coincident.
        if length_sqr == 0.0 {
            let mut rng = rand::rng();

            component_form.x += rng.random_range::<f32, Range<f32>>(-1.0_f32..1.0_f32);
            component_form.y += rng.random_range::<f32, Range<f32>>(-1.0_f32..1.0_f32);
            length_sqr = component_form.length_squared();
            if length_sqr < 1.0 {
                length_sqr = f32::sqrt(length_sqr);
            }
        }

        let f = repel_force * (mass1 * mass2).abs() / length_sqr;
        let dir_vec_normalized = component_form.normalize_or(Vec2::ZERO);
        dir_vec_normalized * f
    }
}

#[cfg(test)]
mod test {
    #![allow(clippy::unwrap_used, reason = "Testing is allowed to panic")]

    use super::*;

    #[test]
    fn test_bounding_box_section() {
        let bb: BoundingBox2D = BoundingBox2D::new(Vec2::ZERO, 10.0, 10.0);
        assert_eq!(bb.section(Vec2::new(-1.0, -1.0)), 0);
        assert_eq!(bb.section(Vec2::new(1.0, -1.0)), 1);
        assert_eq!(bb.section(Vec2::new(-1.0, 1.0)), 2);
        assert_eq!(bb.section(Vec2::new(1.0, 1.0)), 3);
    }

    #[test]
    fn test_bounding_box_sub_quadrant() {
        let bb: BoundingBox2D = BoundingBox2D::new(Vec2::ZERO, 10.0, 10.0);
        assert_eq!(
            bb.sub_quadrant(0),
            BoundingBox2D::new(Vec2::new(-2.5, -2.5), 5.0, 5.0)
        );
        assert_eq!(
            bb.sub_quadrant(1),
            BoundingBox2D::new(Vec2::new(2.5, -2.5), 5.0, 5.0)
        );
        assert_eq!(
            bb.sub_quadrant(2),
            BoundingBox2D::new(Vec2::new(-2.5, 2.5), 5.0, 5.0)
        );
        assert_eq!(
            bb.sub_quadrant(3),
            BoundingBox2D::new(Vec2::new(2.5, 2.5), 5.0, 5.0)
        );
    }

    #[test]
    #[expect(clippy::float_cmp)]
    fn test_quadtree_insert() {
        let mut qt: QuadTree = QuadTree::new(BoundingBox2D::new(Vec2::ZERO, 10.0, 10.0));
        // Insert first node
        let n1_mass = 5.0;
        qt.insert(Vec2::new(-1.0, -1.0), n1_mass).unwrap();
        if let Node::Leaf { mass, .. } = qt.children[&0] {
            assert_eq!(mass, n1_mass);
        } else {
            panic!("New node should be leaf")
        }

        // Insert second node in in the same quadrant but different sub quadrant
        //  N1-R-N2
        let n2_mass = 30.0;
        qt.insert(Vec2::new(1.0, 1.0), n2_mass).unwrap();
        // check root node
        assert!(qt.children[&0].is_root());
        if let Node::Root { indices, mass, .. } = qt.children[&0] {
            assert_eq!(mass, n1_mass + n2_mass);

            // check node0
            assert_eq!(indices[0], 2);
            assert!(qt.children[&1].is_leaf());

            // check node1
            assert_eq!(indices[3], 1);
            assert!(qt.children[&2].is_leaf());
        }
    }

    #[test]
    fn test_quadtree_contains() {
        let mut qt: QuadTree = QuadTree::new(BoundingBox2D::new(Vec2::ZERO, 10.0, 10.0));
        // Insert first node
        let n1_mass = 5.0;
        let n1_pos = Vec2::new(-1.0, -1.0);
        qt.insert(n1_pos, n1_mass).unwrap();

        // Insert second node
        let n2_mass = 1.0;
        let n2_pos = Vec2::new(-2.0, 1.0);
        qt.insert(n2_pos, n2_mass).unwrap();

        assert!(qt.contains(n1_pos));
        assert!(qt.contains(n2_pos));
    }

    #[test]
    fn test_quadtree_delete() {
        let mut qt: QuadTree = QuadTree::new(BoundingBox2D::new(Vec2::ZERO, 10.0, 10.0));
        // Insert first node
        let n1_mass = 5.0;
        let n1_pos = Vec2::new(-1.0, -1.0);
        qt.insert(n1_pos, n1_mass).unwrap();

        // Insert second node in in the same quadrant but different sub quadrant
        let n2_mass = 30.0;
        let n2_pos = Vec2::new(1.0, 1.0);
        qt.insert(n2_pos, n2_mass).unwrap();

        qt.delete_point(n2_pos).unwrap();
        assert!(qt.leaves() == 1);

        qt.delete_point(n1_pos).unwrap();
        assert!(qt.is_empty());
    }

    #[ignore = "Delete doesn't remove all nodes as expected. Don't use delete until this test is passing"]
    #[test]
    fn test_quadtree_insert_delete() {
        const NODES: u32 = 100_000;

        let w = 1000.0;
        let bb = BoundingBox2D::new(Vec2::ZERO, w, w);
        let mut qt: QuadTree = QuadTree::new(bb);
        let mut rng = StdRng::seed_from_u64(42);

        let mut points = Vec::with_capacity(NODES as usize);

        for _ in 0..NODES {
            points.push(Vec2::new(
                rng.random_range((-w / 2.0)..(w / 2.0)),
                rng.random_range((-w / 2.0)..(w / 2.0)),
            ));
        }

        for i in 0..NODES {
            qt.insert_id(i, points[i as usize], rng.random_range(1.0..2000.0))
                .unwrap();
        }

        for point in points.iter().take(NODES as usize) {
            // If points are too close to each other they're merged in the tree.
            // Specifically if `p1.distance_squared(p2) <= EPSILON`.
            // This means that we may not be able to delete the same amount of points as we inserted.
            // Therefore, we silence the error here.
            let _ = qt.delete_point(*point);
        }

        assert!(qt.is_empty());
    }

    #[test]
    fn test_quadtree_barnes_hut() {
        const NODES: u32 = 100_000;
        const THETA: f32 = 1.0;
        const REPEL_FORCE: f32 = -1e8;
        const DELTA_TIME: f32 = 0.01;
        const DAMPING: f32 = 0.9;

        let w = 1000.0;
        let bb = BoundingBox2D::new(Vec2::ZERO, w, w);
        let mut qt: QuadTree = QuadTree::new(bb);
        let mut rng = StdRng::seed_from_u64(42);

        let mut velocities: Vec<Vec2> = Vec::with_capacity(NODES as usize);
        let mut points = Vec::with_capacity(NODES as usize);
        let mut masses = Vec::with_capacity(NODES as usize);

        for _ in 0..NODES {
            points.push(Vec2::new(
                rng.random_range((-w / 2.0)..(w / 2.0)),
                rng.random_range((-w / 2.0)..(w / 2.0)),
            ));
            masses.push(rng.random_range(1.0..2000.0));
            velocities.push(Vec2::ZERO);
        }

        for i in 0..NODES {
            qt.insert_id(i, points[i as usize], masses[i as usize])
                .unwrap();
        }

        // Run Barnes-Hut for 25 iterations
        for _ in 0..25 {
            for i in 0..NODES {
                let mass = masses[i as usize];
                let force =
                    qt.approximate_forces_on_body(i, points[i as usize], mass, THETA, REPEL_FORCE);

                velocities[i as usize] += force / mass * DELTA_TIME * DAMPING;
                points[i as usize] += velocities[i as usize] * DELTA_TIME;
            }
        }
    }
}
