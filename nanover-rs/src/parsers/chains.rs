use crate::parsers::residues::ResidueIterator;
use crate::parsers::MolecularSystem;

pub struct ChainView<'a> {
    system: &'a MolecularSystem,
    pub start_index: usize,
    pub next_index: usize,
}

impl ChainView<'_> {
    pub fn iter_residues(&self) -> ResidueIterator {
        ResidueIterator::new(self.system, self.start_index, self.next_index)
    }
}

pub struct ChainIterator<'a> {
    system: &'a MolecularSystem,
    particle_index: usize,
}

impl<'a> ChainIterator<'a> {
    pub fn new(system: &'a MolecularSystem) -> Self {
        Self {
            system,
            particle_index: 0,
        }
    }
}

impl<'a> Iterator for ChainIterator<'a> {
    type Item = ChainView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.particle_index >= self.system.atom_count() {
            return None;
        }

        let start = self.particle_index;
        let first_residue = self.system.atom_resindex[start];
        let mut current_chain = self.system.residue_chain_index[first_residue];
        let mut i = 0;
        for atom in self.particle_index..=self.system.atom_count() {
            i = atom;
            match self.system.atom_resindex.get(atom) {
                None => break,
                Some(residue) => {
                    let chain = self.system.residue_chain_index[*residue];
                    if chain != current_chain {
                        break;
                    }
                    current_chain = chain;
                }
            }
        }
        self.particle_index = i;

        Some(ChainView {
            system: self.system,
            start_index: start,
            next_index: self.particle_index,
        })
    }
}
