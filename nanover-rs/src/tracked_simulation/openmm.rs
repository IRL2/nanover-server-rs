use std::{
    cmp::Ordering,
    collections::BTreeMap,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use log::info;
use nanover_proto::trajectory::FrameData;

use crate::{
    broadcaster::BroadcastSendError,
    frame_broadcaster::FrameBroadcaster,
    simulation::{CoordMap, OpenMMSimulation, Simulation, ToFrameData, IMD},
    state_broadcaster::StateBroadcaster,
    state_interaction::read_forces,
};

use super::specific::{Configuration, TrackedSimulation};

pub struct TrackedOpenMMSimulation {
    pub simulation: OpenMMSimulation,
    simulation_frame: usize,
    reset_counter: usize,
    simulation_counter: usize,
    user_forces: CoordMap,
    user_energies: f64,
    frame_interval: usize,
}

impl TrackedOpenMMSimulation {
    pub fn new(
        simulation: OpenMMSimulation,
        simulation_counter: usize,
        frame_interval: usize,
    ) -> Self {
        let user_forces = BTreeMap::new();
        let user_energies = 0.0;
        let reset_counter = 0;
        let simulation_frame = 0;
        Self {
            simulation,
            simulation_frame,
            reset_counter,
            simulation_counter,
            user_forces,
            user_energies,
            frame_interval,
        }
    }

    pub fn get_platform_name(&self) -> String {
        self.simulation.get_platform_name()
    }

    fn get_total_energy(&self) -> f64 {
        self.simulation.get_total_energy()
    }

    fn user_forces(&self) -> &CoordMap {
        &self.user_forces
    }

    fn update_user_forces(&mut self, force_map: CoordMap) {
        self.user_forces = force_map;
    }

    fn user_energies(&self) -> f64 {
        self.user_energies
    }

    fn update_user_energies(&mut self, user_energies: f64) {
        self.user_energies = user_energies;
    }

    fn simulation_mut(&mut self) -> &mut OpenMMSimulation {
        &mut self.simulation
    }

    pub fn simulation_loop_iteration(
        &mut self,
        sim_clone: &Arc<Mutex<FrameBroadcaster>>,
        state_clone: &Arc<Mutex<StateBroadcaster>>,
        simulation_tx: &std::sync::mpsc::Sender<usize>,
        now: &Instant,
        configuration: &Configuration,
    ) {
        let (delta_frames, do_frames, do_forces) = next_stop(
            self.simulation_frame,
            configuration.frame_interval,
            configuration.force_interval,
        );
        self.simulation.step(delta_frames as i32);

        if do_forces {
            let mut_simulation = self.simulation_mut();
            let (force_map, user_energies) =
                apply_forces(state_clone, mut_simulation, simulation_tx.clone());
            self.update_user_forces(force_map);
            self.update_user_energies(user_energies);
        }

        let system_energy = self.get_total_energy();

        if do_frames
            && self
                .send_regular_frame(
                    Arc::clone(sim_clone),
                    configuration.with_velocities,
                    configuration.with_forces,
                )
                .is_err()
        {
            return;
        };

        let elapsed = now.elapsed();
        let time_left = match configuration.simulation_interval.checked_sub(elapsed) {
            Some(d) => d,
            None => Duration::from_millis(0),
        };
        if configuration.verbose {
            info!(
                "Simulation frame {}. Time to sleep {time_left:?}. Total energy {system_energy:.2} kJ/mol.",
                self.simulation_frame,
            );
        };
        if configuration.auto_reset && !system_energy.is_finite() {
            self.simulation.reset();
        }
        thread::sleep(time_left);
    }
}

impl TrackedSimulation for TrackedOpenMMSimulation {
    fn step(&mut self) {
        let delta_frames = next_frame_stop(self.simulation_frame, self.frame_interval);
        self.simulation.step(delta_frames as i32);
        self.simulation_frame += delta_frames;
    }

    fn reset(&mut self) {
        self.simulation.reset();
        self.reset_counter += 1;
    }

    fn reset_counter(&self) -> usize {
        self.reset_counter
    }

    fn simulation_counter(&self) -> usize {
        self.simulation_counter
    }

    fn send_regular_frame(
        &mut self,
        sim_clone: Arc<Mutex<FrameBroadcaster>>,
        with_velocities: bool,
        with_forces: bool,
    ) -> Result<(), BroadcastSendError> {
        let mut frame = self.simulation.to_framedata(with_velocities, with_forces);
        let energy_total = self.user_energies();
        frame
            .insert_number_value("energy.user.total", energy_total)
            .unwrap();
        frame
            .insert_number_value("system.reset.counter", self.reset_counter as f64)
            .unwrap();
        add_force_map_to_frame(self.user_forces(), &mut frame);
        let mut source = sim_clone.lock().unwrap();
        source.send_frame(frame)
    }

    fn clear_state(&mut self, _state_clone: Arc<Mutex<StateBroadcaster>>) {}
}

impl ToFrameData for TrackedOpenMMSimulation {
    fn to_framedata(&self, with_velocity: bool, with_forces: bool) -> FrameData {
        self.simulation.to_framedata(with_velocity, with_forces)
    }

    fn to_topology_framedata(&self) -> FrameData {
        self.simulation.to_topology_framedata()
    }
}

fn next_stop(
    current_frame: usize,
    frame_interval: usize,
    force_interval: usize,
) -> (usize, bool, bool) {
    let next_frame_stop = frame_interval - current_frame % frame_interval;
    let next_force_stop = force_interval - current_frame % force_interval;
    match next_frame_stop.cmp(&next_force_stop) {
        Ordering::Less => (next_frame_stop, true, false),
        Ordering::Greater => (next_force_stop, false, true),
        Ordering::Equal => (next_frame_stop, true, true),
    }
}

fn next_frame_stop(current_frame: usize, frame_interval: usize) -> usize {
    frame_interval - current_frame % frame_interval
}

fn apply_forces(
    state_clone: &Arc<Mutex<StateBroadcaster>>,
    simulation: &mut OpenMMSimulation,
    simulation_tx: std::sync::mpsc::Sender<usize>,
) -> (CoordMap, f64) {
    let state_interactions = read_forces(state_clone);
    let imd_interactions = simulation.compute_forces(&state_interactions);
    simulation_tx.send(imd_interactions.len()).unwrap();
    let forces = simulation.update_imd_forces(&imd_interactions).unwrap();
    let interactions = imd_interactions
        .into_iter()
        .filter_map(|interaction| interaction.energy)
        .sum();
    (forces, interactions)
}

fn add_force_map_to_frame(force_map: &CoordMap, frame: &mut FrameData) {
    let indices = force_map.keys().map(|index| *index as u32).collect();
    let sparse = force_map
        .values()
        .flatten()
        .map(|value| *value as f32)
        .collect();
    frame
        .insert_index_array("forces.user.index", indices)
        .unwrap();
    frame
        .insert_float_array("forces.user.sparse", sparse)
        .unwrap();
}
