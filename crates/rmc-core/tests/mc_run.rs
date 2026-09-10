use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use rmc_core::mc::{
    run_chain, Measurement, MetropolisKernel, NoopCallbacks, RunCallbacks, SimulationCtx,
    SimulationParams, SingleUpdateSet, SteppingUpdateSet, Update, UpdateSet, WeightedUpdate,
    WeightedUpdateSet, WithDeadline,
};
use rmc_core::random::{ChainId, SeedSource};

#[derive(Clone)]
struct IncrementUpdate {
    value: Rc<Cell<i32>>,
}

impl Update<()> for IncrementUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut (), _rng: &mut R) -> f64 {
        1.0
    }

    fn accept(&mut self, _state: &mut ()) {
        self.value.set(self.value.get() + 1);
    }
}

#[derive(Clone)]
struct CounterMeasurement {
    count: Rc<Cell<i32>>,
}

impl Measurement<()> for CounterMeasurement {
    type Output = ();

    fn measure(&mut self, _state: &()) {
        self.count.set(self.count.get() + 1);
    }

    fn finish(self) -> Self::Output {}
}

#[derive(Clone)]
struct TypedCounterMeasurement {
    count: u64,
}

impl Measurement<()> for TypedCounterMeasurement {
    type Output = u64;

    fn measure(&mut self, _state: &()) {
        self.count += 1;
    }

    fn finish(self) -> Self::Output {
        self.count
    }
}

impl Measurement<i64> for TypedCounterMeasurement {
    type Output = u64;

    fn measure(&mut self, _state: &i64) {
        self.count += 1;
    }

    fn finish(self) -> Self::Output {
        self.count
    }
}

#[test]
fn run_chain_executes_metropolis_kernel() {
    let value = Rc::new(Cell::new(0));
    let measurement_count = Rc::new(Cell::new(0));
    let updates = SingleUpdateSet::new(IncrementUpdate {
        value: Rc::clone(&value),
    });
    let mut kernel = MetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));
    let (_state, stats, ()) = run_chain(
        (),
        &mut rng,
        &mut kernel,
        CounterMeasurement {
            count: Rc::clone(&measurement_count),
        },
        SimulationParams {
            max_steps: 10,
            steps_per_cycle: 2,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(value.get(), 10);
    assert_eq!(measurement_count.get(), 5);
    assert_eq!(stats.steps_done, 10);
    assert_eq!(stats.cycles_done, 5);

    assert_eq!(kernel.updates().stats()[0].nprops, 10);
    assert_eq!(kernel.updates().stats()[0].naccs, 10);
    assert_eq!(kernel.updates().stats()[0].nimps, 0);
}

#[derive(Default)]
struct StopAfterTwoCycles;

impl RunCallbacks<SimulationCtx> for StopAfterTwoCycles {
    fn stop_when(&mut self, ctx: &SimulationCtx) -> bool {
        ctx.cycles_done >= 2
    }
}

#[test]
fn run_callbacks_can_stop_the_loop() {
    let value = Rc::new(Cell::new(0));
    let updates = SingleUpdateSet::new(IncrementUpdate {
        value: Rc::clone(&value),
    });
    let mut kernel = MetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));
    let mut callbacks = StopAfterTwoCycles;

    let (_state, stats, ()) = run_chain(
        (),
        &mut rng,
        &mut kernel,
        CounterMeasurement {
            count: Rc::new(Cell::new(0)),
        },
        SimulationParams {
            max_steps: 100,
            steps_per_cycle: 3,
            cycles_per_check: 1,
        },
        &mut callbacks,
    )
    .unwrap();

    assert_eq!(stats.steps_done, 6);
    assert_eq!(stats.cycles_done, 2);
    assert_eq!(value.get(), 6);
}

#[test]
fn with_deadline_stops_when_past_and_delegates_when_none() {
    let ctx = SimulationCtx {
        steps_done: 0,
        cycles_done: 0,
        steps_in_cycle: 0,
    };

    let mut expired = WithDeadline {
        inner: NoopCallbacks,
        deadline: Some(Instant::now() - Duration::from_secs(1)),
    };
    assert!(expired.stop_when(&ctx));

    let mut inert = WithDeadline {
        inner: NoopCallbacks,
        deadline: None,
    };
    assert!(!inert.stop_when(&ctx));

    // A None deadline still forwards the inner callback's stop decision.
    let mut delegating = WithDeadline {
        inner: StopAfterTwoCycles,
        deadline: None,
    };
    assert!(delegating.stop_when(&SimulationCtx {
        cycles_done: 2,
        ..ctx
    }));
}

#[test]
fn run_chain_returns_measurement_output_by_ownership() {
    let value = Rc::new(Cell::new(0));
    let updates = SingleUpdateSet::new(IncrementUpdate {
        value: Rc::clone(&value),
    });
    let mut kernel = MetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let (_state, stats, measurement_count) = run_chain(
        (),
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 7,
            steps_per_cycle: 3,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(value.get(), 7);
    assert_eq!(stats.steps_done, 7);
    assert_eq!(stats.cycles_done, 3);
    assert_eq!(measurement_count, 3);
}

#[test]
fn run_rejects_zero_steps_per_cycle() {
    let updates = SingleUpdateSet::new(IncrementUpdate {
        value: Rc::new(Cell::new(0)),
    });
    let mut kernel = MetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let err = run_chain(
        (),
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 10,
            steps_per_cycle: 0,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )
    .unwrap_err();

    assert_eq!(
        err.to_string(),
        "invalid argument: steps_per_cycle must be > 0"
    );
}

#[derive(Clone)]
struct CountingUpdate {
    accepted: Rc<Cell<i32>>,
    rejected: Rc<Cell<i32>>,
    probability: f64,
}

impl Update<()> for CountingUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut (), _rng: &mut R) -> f64 {
        self.probability
    }

    fn accept(&mut self, _state: &mut ()) {
        self.accepted.set(self.accepted.get() + 1);
    }

    fn reject(&mut self, _state: &mut ()) {
        self.rejected.set(self.rejected.get() + 1);
    }
}

#[test]
fn run_chain_works_with_static_single_update_set() {
    let accepted = Rc::new(Cell::new(0));
    let rejected = Rc::new(Cell::new(0));
    let updates = SingleUpdateSet::new(CountingUpdate {
        accepted: Rc::clone(&accepted),
        rejected: Rc::clone(&rejected),
        probability: 1.0,
    });
    let mut kernel = MetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let (_state, stats, measurement_count) = run_chain(
        (),
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 8,
            steps_per_cycle: 4,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(accepted.get(), 8);
    assert_eq!(rejected.get(), 0);
    assert_eq!(stats.steps_done, 8);
    assert_eq!(measurement_count, 2);
    assert_eq!(kernel.updates().stats()[0].nprops, 8);
    assert_eq!(kernel.updates().stats()[0].naccs, 8);
    assert_eq!(kernel.updates().stats()[0].nimps, 0);
}

#[test]
fn static_single_update_set_tracks_rejections_and_impossible_updates() {
    let accepted = Rc::new(Cell::new(0));
    let rejected = Rc::new(Cell::new(0));
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let mut rejected_set = SingleUpdateSet::new(CountingUpdate {
        accepted: Rc::clone(&accepted),
        rejected: Rc::clone(&rejected),
        probability: 0.0,
    });
    let rejected_outcome = rejected_set.select_and_step(&mut (), &mut rng).unwrap();
    assert!(!rejected_outcome.accepted);
    assert!(!rejected_outcome.impossible);
    assert_eq!(accepted.get(), 0);
    assert_eq!(rejected.get(), 1);
    assert_eq!(rejected_set.stats()[0].nprops, 1);
    assert_eq!(rejected_set.stats()[0].naccs, 0);
    assert_eq!(rejected_set.stats()[0].nimps, 0);

    let mut impossible_set = SingleUpdateSet::new(CountingUpdate {
        accepted: Rc::clone(&accepted),
        rejected: Rc::clone(&rejected),
        probability: -1.0,
    });
    let impossible_outcome = impossible_set.select_and_step(&mut (), &mut rng).unwrap();
    assert!(!impossible_outcome.accepted);
    assert!(impossible_outcome.impossible);
    assert_eq!(accepted.get(), 0);
    assert_eq!(rejected.get(), 2);
    assert_eq!(impossible_set.stats()[0].nprops, 1);
    assert_eq!(impossible_set.stats()[0].naccs, 0);
    assert_eq!(impossible_set.stats()[0].nimps, 1);
}

#[test]
fn weighted_update_set_skips_zero_weight_update() {
    let first_accepted = Rc::new(Cell::new(0));
    let first_rejected = Rc::new(Cell::new(0));
    let second_accepted = Rc::new(Cell::new(0));
    let second_rejected = Rc::new(Cell::new(0));

    let updates = WeightedUpdateSet::new(vec![
        WeightedUpdate::new(
            CountingUpdate {
                accepted: Rc::clone(&first_accepted),
                rejected: Rc::clone(&first_rejected),
                probability: 1.0,
            },
            0.0,
        ),
        WeightedUpdate::new(
            CountingUpdate {
                accepted: Rc::clone(&second_accepted),
                rejected: Rc::clone(&second_rejected),
                probability: 1.0,
            },
            1.0,
        ),
    ])
    .unwrap();
    let mut kernel = MetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let (_state, stats, measurement_count) = run_chain(
        (),
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 6,
            steps_per_cycle: 2,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(stats.steps_done, 6);
    assert_eq!(measurement_count, 3);
    assert_eq!(first_accepted.get(), 0);
    assert_eq!(second_accepted.get(), 6);
    assert_eq!(kernel.updates().stats()[0].nprops, 0);
    assert_eq!(kernel.updates().stats()[1].nprops, 6);
    assert_eq!(kernel.updates().stats()[1].naccs, 6);
}

#[test]
fn weighted_update_set_runs_many_updates_of_one_type() {
    let accepted = Rc::new(Cell::new(0));
    let rejected = Rc::new(Cell::new(0));
    let updates = WeightedUpdateSet::new(vec![
        WeightedUpdate::new(
            CountingUpdate {
                accepted: Rc::clone(&accepted),
                rejected: Rc::clone(&rejected),
                probability: 1.0,
            },
            0.0,
        ),
        WeightedUpdate::new(
            CountingUpdate {
                accepted: Rc::clone(&accepted),
                rejected: Rc::clone(&rejected),
                probability: 1.0,
            },
            1.0,
        ),
        WeightedUpdate::new(
            CountingUpdate {
                accepted: Rc::clone(&accepted),
                rejected: Rc::clone(&rejected),
                probability: 1.0,
            },
            0.0,
        ),
    ])
    .unwrap();
    let mut kernel = MetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let (_state, stats, measurement_count) = run_chain(
        (),
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 5,
            steps_per_cycle: 1,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(stats.steps_done, 5);
    assert_eq!(measurement_count, 5);
    assert_eq!(accepted.get(), 5);
    assert_eq!(rejected.get(), 0);
    assert_eq!(kernel.updates().stats()[0].nprops, 0);
    assert_eq!(kernel.updates().stats()[1].nprops, 5);
    assert_eq!(kernel.updates().stats()[2].nprops, 0);
}

#[derive(Clone)]
struct AddToState(i64);

impl Update<i64> for AddToState {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut i64, _rng: &mut R) -> f64 {
        1.0
    }

    fn accept(&mut self, state: &mut i64) {
        *state += self.0;
    }
}

#[derive(Clone)]
struct MultiplyState(i64);

impl Update<i64> for MultiplyState {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut i64, _rng: &mut R) -> f64 {
        1.0
    }

    fn accept(&mut self, state: &mut i64) {
        *state *= self.0;
    }
}

#[derive(Clone)]
enum ToyEnumUpdate {
    Add(AddToState),
    Multiply(MultiplyState),
}

impl Update<i64> for ToyEnumUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut i64, rng: &mut R) -> f64 {
        match self {
            Self::Add(update) => update.attempt(state, rng),
            Self::Multiply(update) => update.attempt(state, rng),
        }
    }

    fn accept(&mut self, state: &mut i64) {
        match self {
            Self::Add(update) => update.accept(state),
            Self::Multiply(update) => update.accept(state),
        }
    }

    fn reject(&mut self, state: &mut i64) {
        match self {
            Self::Add(update) => update.reject(state),
            Self::Multiply(update) => update.reject(state),
        }
    }
}

#[test]
fn weighted_update_set_supports_enum_wrapped_heterogeneous_updates() {
    let updates = WeightedUpdateSet::new(vec![
        WeightedUpdate::new(ToyEnumUpdate::Add(AddToState(3)), 1.0),
        WeightedUpdate::new(ToyEnumUpdate::Multiply(MultiplyState(10)), 0.0),
    ])
    .unwrap();
    let mut kernel = MetropolisKernel::new(updates);
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));

    let (state, stats, measurement_count) = run_chain(
        0_i64,
        &mut rng,
        &mut kernel,
        TypedCounterMeasurement { count: 0 },
        SimulationParams {
            max_steps: 4,
            steps_per_cycle: 2,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(state, 12);
    assert_eq!(stats.steps_done, 4);
    assert_eq!(measurement_count, 2);
    assert_eq!(kernel.updates().stats()[0].nprops, 4);
    assert_eq!(kernel.updates().stats()[1].nprops, 0);
}

#[test]
fn weighted_update_set_inverse_pair_sets_ratios() {
    let updates = WeightedUpdateSet::inverse_pair(
        CountingUpdate {
            accepted: Rc::new(Cell::new(0)),
            rejected: Rc::new(Cell::new(0)),
            probability: 0.5,
        },
        2.0,
        CountingUpdate {
            accepted: Rc::new(Cell::new(0)),
            rejected: Rc::new(Cell::new(0)),
            probability: 0.5,
        },
        4.0,
    )
    .unwrap();

    assert_eq!(updates.weights(), vec![2.0, 4.0]);
    assert_eq!(updates.ratios(), vec![2.0, 0.5]);
}

#[test]
fn weighted_update_set_validates_weights() {
    let accepted = Rc::new(Cell::new(0));
    let rejected = Rc::new(Cell::new(0));
    let update = || CountingUpdate {
        accepted: Rc::clone(&accepted),
        rejected: Rc::clone(&rejected),
        probability: 1.0,
    };

    assert!(WeightedUpdateSet::new(Vec::<WeightedUpdate<CountingUpdate>>::new()).is_err());
    assert!(WeightedUpdateSet::new(vec![WeightedUpdate::new(update(), 0.0)]).is_err());
    assert!(WeightedUpdateSet::new(vec![WeightedUpdate::new(update(), -1.0)]).is_err());
    assert!(WeightedUpdateSet::new(vec![
        WeightedUpdate::new(update(), 0.0),
        WeightedUpdate::new(update(), 0.0),
    ])
    .is_err());
    assert!(WeightedUpdateSet::new(vec![
        WeightedUpdate::new(update(), -1.0),
        WeightedUpdate::new(update(), 1.0),
    ])
    .is_err());
    assert!(WeightedUpdateSet::inverse_pair(update(), 0.0, update(), 1.0).is_err());
    assert!(WeightedUpdateSet::inverse_pair(update(), 0.0, update(), 0.0).is_err());
}

#[derive(Clone)]
struct RandomWalkUpdate {
    position: Rc<Cell<i32>>,
    pending_delta: i32,
}

impl RandomWalkUpdate {
    fn new(position: Rc<Cell<i32>>) -> Self {
        Self {
            position,
            pending_delta: 0,
        }
    }
}

impl Update<()> for RandomWalkUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut (), rng: &mut R) -> f64 {
        self.pending_delta = if rng.next_u64() & 1 == 0 { 1 } else { -1 };
        1.0
    }

    fn accept(&mut self, _state: &mut ()) {
        self.position.set(self.position.get() + self.pending_delta);
    }
}

#[test]
fn update_attempt_can_draw_from_chain_rng() {
    fn run_walk(seed: u64) -> i32 {
        let position = Rc::new(Cell::new(0));
        let updates = SingleUpdateSet::new(RandomWalkUpdate::new(Rc::clone(&position)));
        let mut kernel = MetropolisKernel::new(updates);
        let mut rng = SeedSource::new(seed).rng_for(ChainId(0));

        run_chain(
            (),
            &mut rng,
            &mut kernel,
            TypedCounterMeasurement { count: 0 },
            SimulationParams {
                max_steps: 64,
                steps_per_cycle: 8,
                cycles_per_check: 1,
            },
            NoopCallbacks,
        )
        .unwrap();

        position.get()
    }

    assert_eq!(run_walk(123), run_walk(123));
}
