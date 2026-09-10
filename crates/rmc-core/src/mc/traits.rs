use rand::Rng;

use crate::Result;

/// A single Monte Carlo update acting on a chain's `State`.
///
/// This is the one update trait: there is no separate "stateless" variant. A stateless update is
/// simply `Update<()>` — its `state` argument is the zero-sized unit and carries no information.
/// Owning the mutated data in `State` (threaded by the run loop) is preferred over reaching for
/// `Arc`/`Rc` shared cells, because each parallel chain then owns an independent `State`.
///
/// `attempt` is generic over the concrete RNG (`R: Rng + ?Sized`) so every `rng.next_u64()` /
/// `rng.gen()` call inside an update can inline instead of going through a vtable.
pub trait Update<State> {
    /// Propose a move and return its acceptance probability.
    ///
    /// A returned value `< 0.0` marks the move as *impossible* (rejected, counted separately);
    /// `>= 1.0` is always accepted; otherwise it is accepted with the returned probability.
    fn attempt<R: Rng + ?Sized>(&mut self, state: &mut State, rng: &mut R) -> f64;

    /// Commit the proposed move to `state`.
    fn accept(&mut self, state: &mut State);

    /// Roll back the proposed move. Defaults to a no-op (the common case where `attempt` does not
    /// mutate `state` until `accept`).
    fn reject(&mut self, _state: &mut State) {}
}

/// Define an enum wrapper that dispatches [`Update`] to its variants.
///
/// Add `; reject` when the wrapped updates need rollback forwarding. Without it, the generated
/// impl uses [`Update::reject`]'s default no-op.
#[macro_export]
macro_rules! dispatch_update {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident <$state:ty> {
            $($variant:ident($update:ty) $(=> $label:expr)?),+ $(,)?
        }
        $(; $reject:ident)?
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $($variant($update)),+
        }

        impl $name {
            pub fn name(&self) -> &'static str {
                match self {
                    $(Self::$variant(_) => $crate::dispatch_update!(@name $variant $($label)?)),+
                }
            }
        }

        impl $crate::mc::Update<$state> for $name {
            fn attempt<R: $crate::__rand::Rng + ?Sized>(
                &mut self,
                state: &mut $state,
                rng: &mut R,
            ) -> f64 {
                match self {
                    $(Self::$variant(update) => update.attempt(state, rng)),+
                }
            }

            fn accept(&mut self, state: &mut $state) {
                match self {
                    $(Self::$variant(update) => update.accept(state)),+
                }
            }

            $crate::dispatch_update!(@reject [$state] [$($variant),+] $($reject)?);
        }
    };
    (@reject [$state:ty] [$($variant:ident),+] reject) => {
        fn reject(&mut self, state: &mut $state) {
            match self {
                $(Self::$variant(update) => update.reject(state)),+
            }
        }
    };
    (@reject [$state:ty] [$($variant:ident),+]) => {};
    (@name $variant:ident $label:expr) => {
        $label
    };
    (@name $variant:ident) => {
        stringify!($variant)
    };
}

/// A per-cycle measurement of a chain's `State`, returning a typed `Output` by ownership.
///
/// A stateless measurement is `Measurement<()>`. Results flow back through `finish` with no
/// `Any`/downcast; see [`run_chain`](crate::mc::run_chain). Several measurements can be composed
/// as a tuple.
pub trait Measurement<State> {
    type Output;

    fn measure(&mut self, state: &State);
    fn finish(self) -> Self::Output;
}

macro_rules! impl_measurement_tuple {
    ($($m:ident $idx:tt),+) => {
        impl<State, $($m: Measurement<State>),+> Measurement<State> for ($($m,)+) {
            type Output = ($($m::Output,)+);

            fn measure(&mut self, state: &State) {
                $(self.$idx.measure(state);)+
            }

            fn finish(self) -> Self::Output {
                ($(self.$idx.finish(),)+)
            }
        }
    };
}

impl_measurement_tuple!(M0 0, M1 1);
impl_measurement_tuple!(M0 0, M1 1, M2 2);
impl_measurement_tuple!(M0 0, M1 1, M2 2, M3 3);

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOutcome {
    pub update_index: usize,
    pub probability: f64,
    pub accepted: bool,
    pub impossible: bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UpdateStats {
    pub nprops: u64,
    pub naccs: u64,
    pub nimps: u64,
}

/// Common, dispatch-free view over an update set's per-update statistics.
pub trait UpdateSet {
    fn stats(&self) -> &[UpdateStats];
    fn reset_stats(&mut self);
}

/// An update set that can be driven for one MC step against a chain's `State`.
///
/// The static implementors ([`SingleUpdateSet`](crate::mc::SingleUpdateSet),
/// [`WeightedUpdateSet`](crate::mc::WeightedUpdateSet))
/// are monomorphized over `R`, so RNG draws inline.
pub trait SteppingUpdateSet<State, R>: UpdateSet {
    fn prepare(&mut self, state: &mut State) -> Result<()>;
    fn select_and_step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome>;
}

/// A kernel advances a chain by one MC step. A stateless kernel is `Kernel<(), R>`.
pub trait Kernel<State, R> {
    fn prepare(&mut self, _state: &mut State) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome>;

    fn reset_stats(&mut self) {}
}

pub trait RunCallbacks<C> {
    fn on_step(&mut self, _ctx: &C) {}
    fn on_cycle(&mut self, _ctx: &C) {}
    fn on_checkpoint(&mut self, _ctx: &C) {}
    fn stop_when(&mut self, _ctx: &C) -> bool {
        false
    }
}

/// Callback wrapper that adds a wall-clock deadline to `stop_when`.
///
/// `Option<Instant>` is `Copy`, so every chain's callback carries the same deadline value with no
/// shared state. `None` delegates to `inner` unchanged.
#[derive(Clone, Copy, Debug)]
pub struct WithDeadline<C> {
    pub inner: C,
    pub deadline: Option<std::time::Instant>,
}

impl<Ctx, C: RunCallbacks<Ctx>> RunCallbacks<Ctx> for WithDeadline<C> {
    fn on_step(&mut self, ctx: &Ctx) {
        self.inner.on_step(ctx)
    }

    fn on_cycle(&mut self, ctx: &Ctx) {
        self.inner.on_cycle(ctx)
    }

    fn on_checkpoint(&mut self, ctx: &Ctx) {
        self.inner.on_checkpoint(ctx)
    }

    fn stop_when(&mut self, ctx: &Ctx) -> bool {
        self.inner.stop_when(ctx)
            || self
                .deadline
                .is_some_and(|deadline| std::time::Instant::now() >= deadline)
    }
}

impl<Ctx, C: RunCallbacks<Ctx> + ?Sized> RunCallbacks<Ctx> for &mut C {
    fn on_step(&mut self, ctx: &Ctx) {
        (**self).on_step(ctx)
    }

    fn on_cycle(&mut self, ctx: &Ctx) {
        (**self).on_cycle(ctx)
    }

    fn on_checkpoint(&mut self, ctx: &Ctx) {
        (**self).on_checkpoint(ctx)
    }

    fn stop_when(&mut self, ctx: &Ctx) -> bool {
        (**self).stop_when(ctx)
    }
}
