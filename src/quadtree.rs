use glam::Vec2;
use log::debug;
use smallvec::{SmallVec, smallvec};
use std::mem::{swap, take};

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
        parent: u32,
    },
    Leaf {
        mass: f32,
        pos: Vec2,
        parent: u32,
    },
}

impl Node {
    #[must_use]
    const fn new_leaf(pos: Vec2, mass: f32, parent: u32) -> Self {
        Self::Leaf { mass, pos, parent }
    }
    #[must_use]
    const fn new_root(pos: Vec2, mass: f32, indices: [u32; 4], parent: u32) -> Self {
        Self::Root {
            indices,
            mass,
            pos,
            parent,
        }
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
}

#[derive(Debug, Default)]
/// A quadtree capable of storing [`u32::MAX`] elements.
pub struct QuadTree {
    children: Vec<Node>,
    pub boundary: BoundingBox2D,
    root: u32,
}

impl QuadTree {
    #[must_use]
    pub const fn new(boundary: BoundingBox2D) -> Self {
        Self {
            root: 0,
            boundary,
            children: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_capacity(boundary: BoundingBox2D, capacity: usize) -> Self {
        Self {
            root: 0,
            boundary,
            children: Vec::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, new_pos: Vec2, new_mass: f32) -> Result<(), String> {
        if self.children.len() == (UNINITIALIZED - 1) as usize {
            return Err(format!(
                "Quadtree is full! (storing {}/{} elements)",
                self.children.len(),
                UNINITIALIZED - 1
            ));
        }

        self.children
            .push(Node::new_leaf(new_pos, new_mass, self.root));

        // With only one node there's no need to continue
        if self.children.len() == 1 {
            return Ok(());
        }

        let mut bb = self.boundary.clone();
        let mut root_index = self.root;
        let new_index = self.children.len() as u32 - 1;

        // Traversing the tree until we find an empty quadrant for the new leaf.
        while let Node::Root {
            indices,
            mass,
            pos,
            parent,
        } = &mut self.children[root_index as usize]
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

        // if new leaf is too close to current leaf we merge
        // TODO: in this case we will have a "dead" leaf
        if let Node::Leaf { mass, pos, parent } = self.children[root_index as usize]
            && pos.distance(new_pos) < EPSILON
        {
            let m: f32 = mass + new_mass;
            self.children[root_index as usize] = Node::new_leaf(pos, m, parent);
            return Ok(());
        }

        // create new root until leaf and new leaf are in different sections
        while let Node::Leaf { mass, pos, parent } = self.children[root_index as usize] {
            let mut fin = false;

            // Pushes the old leaf to the back of the vector and inserts its index into the index array of the new root
            let old_node = Node::new_leaf(pos, mass, root_index);
            self.children.push(old_node);
            let old_index = self.children.len() - 1;
            let section = bb.section(pos);
            let mut ind = [UNINITIALIZED, UNINITIALIZED, UNINITIALIZED, UNINITIALIZED];
            ind[section as usize] = old_index as u32;

            let section = bb.section(new_pos);

            // If section of the new root is empty we can set it and exit
            if ind[section as usize] == UNINITIALIZED {
                ind[section as usize] = new_index;
                fin = true;
            }

            // sets the old leaf index to the new root
            let new_root = Node::new_root(
                pos * mass + new_pos * new_mass,
                mass + new_mass,
                ind,
                parent,
            );
            self.children[root_index as usize] = new_root;

            if fin {
                break;
            }

            root_index = old_index as u32;

            bb = bb.sub_quadrant(section);
        }

        Ok(())
    }

    /// Removes a node and all its descendants from the quadtree.
    ///
    /// # Panics
    /// Panics if `index` is larger than the size of the quadtree.
    fn delete_index(&mut self, index: u32) {
        // Use of smallvec to avoid heap allocations
        let mut stack: SmallVec<[u32; 4]> = smallvec![index];
        let mut new_stack: SmallVec<[u32; 4]> = SmallVec::with_capacity(4);
        'outer: loop {
            for node_index in &stack {
                // FIXME: Use a hashmap instead of O(n) vector shifting.
                let node = self.children.remove(*node_index as usize);
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
    }

    /// Unassigns the index of a node, making it the uninitialized value.
    ///
    /// If an update leaves all children of a root node uninitialized, the root is converted to a leaf.
    /// This proceeds recursively up towards the root of the quadtree.
    ///
    /// Note that position and mass are unaffected by this method.
    fn unassign_index(&mut self, section: u8, root_index: u32) {
        if let Node::Root {
            indices,
            pos,
            mass,
            parent,
        } = &mut self.children[root_index as usize]
        {
            indices[section as usize] = UNINITIALIZED;

            if indices.iter().all(|i| *i == UNINITIALIZED) {
                // All children are uninitialized. We can convert the node into a leaf.
                self.children[root_index as usize] = Node::new_leaf(*pos, *mass, *parent);
            }
        }

        if let Node::Leaf { parent, .. } = &self.children[root_index as usize] {
            self.unassign_index(section, *parent);
        }
    }

    /// Removes the point `delete_pos` from the quadtree, if it exists.
    pub fn delete_point(&mut self, delete_pos: Vec2) -> Result<(), String> {
        if let Some(delete_node) = self.query_point(delete_pos) {
            let mut bb = self.boundary.clone();
            let mut current_index = self.root;
            let mut parent_index = UNINITIALIZED;
            let del_mas = delete_node.mass();

            // Traversing the tree until we find `delete_pos`.
            while let Node::Root {
                indices,
                mass,
                pos,
                parent,
            } = &mut self.children[current_index as usize]
            {
                let section = bb.section(delete_pos);

                if *pos == delete_pos {
                    parent_index = *parent;
                    break;
                }

                // Update Mass and Pos of root to account for the removed node (when we find it).
                *mass -= del_mas;
                *pos -= delete_pos * del_mas;

                current_index = indices[section as usize];
                bb = bb.sub_quadrant(section);
            }

            // Check the final leaf node
            if let Node::Leaf { mass, pos, parent } = &self.children[current_index as usize]
                && *pos == delete_pos
            {
                parent_index = *parent;
            }

            if parent_index != UNINITIALIZED {
                // Remove the node
                self.delete_index(current_index);

                // Set the deleted node to uninitialized in the parent node.
                let section = bb.section(delete_pos);
                self.unassign_index(section, parent_index);
                return Ok(());
            }
        }
        Err(format!(
            "Failed to delete point '{delete_pos}': point not found"
        ))
    }

    /// Returns the quadtree node having position `query_pos`.
    #[must_use]
    pub fn query_point(&self, query_pos: Vec2) -> Option<&Node> {
        let mut bb = self.boundary.clone();
        let mut root_index = self.root;

        // Traversing the tree until we find `query_pos`.
        while let Node::Root { indices, pos, .. } = &self.children[root_index as usize] {
            if *pos == query_pos {
                return Some(&self.children[root_index as usize]);
            }
            let section = bb.section(query_pos);
            root_index = indices[section as usize];
            bb = bb.sub_quadrant(section);
        }

        // Check the final leaf node
        if let Node::Leaf { pos, .. } = &self.children[root_index as usize]
            && *pos == query_pos
        {
            return Some(&self.children[root_index as usize]);
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

    /// Barnes-Hut algorithm
    ///
    /// `position` is position of the body to compute forces on.
    ///
    /// `mass` is the mass of the body.
    ///
    /// `theta` is the threshold of the Barnes-Hut algorithm.
    ///
    /// `repel_force` is the magnitude of the repulsive force between two bodies (the Coulomb constant).
    ///
    /// Returns the net force acted upon the body for the entire tree.
    #[must_use]
    pub fn barnes_hut(&self, position: Vec2, mass: f32, theta: f32, repel_force: f32) -> Vec2 {
        let mut net_force = Vec2::ZERO;

        if self.children.is_empty() {
            return net_force;
        }

        let mut s: f32 = self.boundary.width.max(self.boundary.height);

        // Use of smallvec to avoid heap allocations
        let mut stack: SmallVec<[u32; 4]> = smallvec![0];
        let mut new_stack: SmallVec<[u32; 4]> = SmallVec::with_capacity(4);
        'outer: loop {
            for node_index in &stack {
                let parent = &self.children[*node_index as usize];

                match parent {
                    Node::Root { indices, .. } => {
                        let center_mass = parent.position();
                        let dist = center_mass.distance(position);
                        if s / dist < theta {
                            net_force += Self::repel_force(
                                position,
                                parent.position(),
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
                    Node::Leaf { .. } => {
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
        let dir_vec = pos2 - pos1;
        let mut length_sqr = dir_vec.length_squared();
        if length_sqr == 0.0 {
            return Vec2::ZERO;
        }

        let f = -repel_force * (mass1 * mass2).abs() / length_sqr;
        let dir_vec_normalized = dir_vec.normalize_or(Vec2::ZERO);
        dir_vec_normalized * f
    }
}

#[cfg(test)]
mod test {
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
        qt.insert(Vec2::new(-1.0, -1.0), n1_mass);
        if let Node::Leaf { mass, .. } = qt.children[0] {
            assert_eq!(mass, n1_mass);
        } else {
            panic!("New node should be leaf")
        }

        // Insert second node in in the same quadrant but different sub quadrant
        //  N1-R-N2
        let n2_mass = 30.0;
        qt.insert(Vec2::new(1.0, 1.0), n2_mass);
        // check root node
        assert!(qt.children[0].is_root());
        if let Node::Root { indices, mass, .. } = qt.children[0] {
            assert_eq!(mass, n1_mass + n2_mass);

            // check node0
            assert_eq!(indices[0], 2);
            assert!(qt.children[1].is_leaf());

            // check node1
            assert_eq!(indices[3], 1);
            assert!(qt.children[2].is_leaf());
        }
    }

    #[test]
    fn test_quadtree_contains() {
        let mut qt: QuadTree = QuadTree::new(BoundingBox2D::new(Vec2::ZERO, 10.0, 10.0));
        // Insert first node
        let n1_mass = 5.0;
        let n1_pos = Vec2::new(-1.0, -1.0);
        qt.insert(n1_pos, n1_mass);

        // Insert second node
        let n2_mass = 1.0;
        let n2_pos = Vec2::new(-2.0, 1.0);
        qt.insert(n2_pos, n2_mass);

        assert!(qt.contains(n1_pos));
        assert!(qt.contains(n2_pos));
    }

    #[ignore = "Test failing due to double insertion bug in QuadTree::insert()"]
    #[test]
    fn test_quadtree_delete() {
        let mut qt: QuadTree = QuadTree::new(BoundingBox2D::new(Vec2::ZERO, 10.0, 10.0));
        // Insert first node
        let n1_mass = 5.0;
        let n1_pos = Vec2::new(-1.0, -1.0);
        qt.insert(n1_pos, n1_mass);

        // Insert second node in in the same quadrant but different sub quadrant
        let n2_mass = 30.0;
        let n2_pos = Vec2::new(1.0, 1.0);
        qt.insert(n2_pos, n2_mass);

        // FIXME: Test failing due to double insertion bug in QuadTree::insert()
        assert!(qt.delete_point(n2_pos).is_ok());
        assert!(qt.children.len() == 1);
        assert!(qt.delete_point(n1_pos).is_ok());
        assert!(qt.children.is_empty());
    }
}
