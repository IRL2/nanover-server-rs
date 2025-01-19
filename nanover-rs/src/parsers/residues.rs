use crate::parsers::MolecularSystem;

pub struct ResidueIterator<'a> {
    system: &'a MolecularSystem,
    particle_index: usize,
    end: usize,
}

impl<'a> ResidueIterator<'a> {
    pub fn new(system: &'a MolecularSystem, start: usize, end: usize) -> Self {
        ResidueIterator {
            system,
            particle_index: start,
            end,
        }
    }
}

impl<'a> Iterator for ResidueIterator<'a> {
    type Item = ResidueView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.end.min(self.system.atom_count());
        if self.particle_index >= end - 1 {
            return None;
        }

        let start = self.particle_index;
        let mut current_residue = self.system.atom_resindex[start];
        let mut i = 0;
        for atom in start..=end {
            i = atom;
            match self.system.atom_resindex.get(atom) {
                None => break,
                Some(residue) if *residue != current_residue => break,
                Some(residue) => current_residue = *residue,
            }
        }

        self.particle_index = i;
        Some(ResidueView {
            system: self.system,
            start_index: start,
            next_index: self.particle_index,
        })
    }
}

pub struct ResidueView<'a> {
    system: &'a MolecularSystem,
    pub start_index: usize,
    pub next_index: usize,
}

impl ResidueView<'_> {
    pub fn find_atom_position(&self, name: &str) -> Option<usize> {
        self.system.names[self.start_index..self.next_index]
            .iter()
            .position(|n| name.trim() == n.trim())
            .map(|position| self.start_index + position)
    }

    pub fn name(&self) -> &str {
        let residue_index = self.system.atom_resindex[self.start_index];
        &self.system.resnames[residue_index]
    }
}
