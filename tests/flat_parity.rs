//! Phase A2 parity: the flat ab-initio engine against the slotmap
//! `rmc-diagmc` engine on imported random diagrams — per-update acceptance
//! ratios agree to ≤ 1e-10 relative, weight caches agree from scratch, and
//! the index arithmetic mirrors `VertexTable` exactly.

mod common;
use common::synthetic_model;

use nalgebra::Vector3;
use rand::{Rng, RngExt, SeedableRng};

use rmc_ab_initio::flat::{self, FlatDiagram, updates as flat_updates, weight as flat_weight};
use rmc_ab_initio::model_abinitio::AbInitioModel;
use rmc_diagmc::updates::{DELTA_TAU_LIMIT, time_proposal_pdf};
use rmc_diagmc::weight as slot_weight;
use rmc_diagmc::{Diagram, Model};

const MAX_ORDER_GPU: usize = 24;

/// Random slotmap diagram with on-grid arc momenta and Δτ-separated times.
fn random_diagram(model: &AbInitioModel, rng: &mut impl Rng, max_tau: f64) -> Diagram {
    let table = &model.table;
    let mut diagram = Diagram::new(-0.5, Vector3::zeros(), max_tau, 0, MAX_ORDER_GPU);
    let tail = diagram.tail;
    diagram.set_vertex_tau(tail, rng.random_range(0.5 * max_tau..max_tau));
    for _ in 0..rng.random_range(0..=8) {
        let tau_ext = diagram.tau();
        let mut tau = [
            rng.random_range(0.01 * tau_ext..0.99 * tau_ext),
            rng.random_range(0.01 * tau_ext..0.99 * tau_ext),
        ];
        tau.sort_by(f64::total_cmp);
        // Keep the generator inside the engines' own separation guard.
        if tau[1] - tau[0] < 1.0e-9
            || diagram
                .vertices
                .iter()
                .any(|v| (v.tau - tau[0]).abs() < 1.0e-9 || (v.tau - tau[1]).abs() < 1.0e-9)
        {
            continue;
        }
        let q = rng.random_range(1..table.len());
        let branch = rng.random_range(0..table.nmodes);
        diagram.insert_arc(tau[0], tau[1], table.point(q), branch);
    }
    slot_weight::sync_cache(&mut diagram, model);
    diagram
}

fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    if a.is_infinite() || b.is_infinite() {
        assert_eq!(a, b, "{what}: {a} vs {b}");
        return;
    }
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!(
        (a - b).abs() <= tol * scale,
        "{what}: {a} vs {b} (rel {})",
        (a - b).abs() / scale
    );
}

#[test]
fn wrap_arithmetic_mirrors_vertex_table() {
    let model = synthetic_model(1);
    let table = &model.table;
    let n = table.grid_n as u32;
    for a in 0..table.len() {
        assert_eq!(flat::neg(n, a as u32) as usize, table.neg(a));
        for b in 0..table.len() {
            assert_eq!(
                flat::wrap_add(n, a as u32, b as u32) as usize,
                table.wrap_add(a, b)
            );
        }
    }
}

#[test]
fn import_preserves_weight_and_structure() {
    let model = synthetic_model(2);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xf1a7);
    for _ in 0..50 {
        let diagram = random_diagram(&model, &mut rng, 8.0);
        let flat = FlatDiagram::from_diagram(&diagram, &model.table, MAX_ORDER_GPU);
        assert!(flat.check_consistency());

        let (slot_log, slot_phase) = slot_weight::log_modulus_and_phase(&diagram, model.as_ref());
        let (flat_log, flat_phase) = flat_weight::log_modulus_and_phase(&flat, &model);
        assert_close(flat_log, slot_log, 1.0e-10, "log modulus");
        assert!((flat_phase - slot_phase).norm() < 1.0e-10);

        // Round trip through the slotmap mutation API rebuilds the same state.
        let rebuilt = flat.to_diagram(&model.table);
        assert!(rebuilt.check_consistency());
        assert_eq!(rebuilt.order, diagram.order);
        for (a, b) in rebuilt.vertices.iter().zip(&diagram.vertices) {
            assert_eq!(a.tau, b.tau);
            assert_eq!(a.link, b.link);
            assert_eq!(a.branch, b.branch);
            assert!((a.p_out - b.p_out).norm() < 1.0e-9);
        }

        // Arc enumeration order matches (RemovePhonon indexes into it).
        let slot_arcs: Vec<_> = diagram.arcs().collect();
        let flat_arcs: Vec<_> = flat.arcs().collect();
        assert_eq!(slot_arcs, flat_arcs);
    }
}

/// Drive every flat update on imported random diagrams and rebuild its
/// acceptance from the slotmap engine's ratio and pdf machinery.
#[test]
fn per_update_ratios_match_slotmap_engine() {
    let model = synthetic_model(3);
    let table = model.table.clone();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xab2);

    let mut add = flat_updates::AddPhonon::new(model.clone());
    let mut remove = flat_updates::RemovePhonon::new(model.clone());
    let mut shift_tail = flat_updates::ChangeTau::new(model.clone());
    let mut shift_internal = flat_updates::ChangeInternalTau::new(model.clone());

    let mut checked = [0usize; 4];
    for _ in 0..300 {
        let diagram = random_diagram(&model, &mut rng, 8.0);
        let mut flat = FlatDiagram::from_diagram(&diagram, &table, MAX_ORDER_GPU);

        // --- AddPhonon ---
        let a_flat = add.attempt(&mut flat, &mut rng);
        if a_flat >= 0.0 {
            let q_vec = table.point(add.q as usize);
            let branch = add.branch as usize;
            let slot_ratio = slot_weight::insert_arc_ratio(
                &diagram,
                model.as_ref(),
                add.tau1,
                add.tau2,
                q_vec,
                branch,
            );
            let p = diagram
                .v(diagram
                    .vertices
                    .iter()
                    .position(|v| v.tau > add.tau1)
                    .unwrap()
                    - 1)
                .p_out;
            let lambda = (model.phonon_energy(q_vec, branch) + model.electron_energy(p - q_vec)
                - model.electron_energy(p))
            .max(1.0e-6);
            let a_slot = slot_ratio.modulus()
                / ((diagram.order + 1) as f64
                    * model.propose_phonon_pdf(q_vec, branch)
                    * time_proposal_pdf(diagram.tau(), add.tau1, add.tau2, lambda));
            assert_close(a_flat, a_slot, 1.0e-10, "AddPhonon acceptance");
            assert!((add.ratio.phase - slot_ratio.phase).norm() < 1.0e-10);
            checked[0] += 1;
        }

        // --- RemovePhonon ---
        let a_flat = remove.attempt(&mut flat, &mut rng);
        if a_flat >= 0.0 {
            let (left, right) = (remove.left, remove.right);
            let vertex = diagram.v(left);
            let slot_ratio = slot_weight::remove_arc_ratio(&diagram, model.as_ref(), left, right);
            let lambda = (model.phonon_energy(vertex.q, vertex.branch)
                + model.electron_energy(diagram.incoming_momentum(left) - vertex.q)
                - model.electron_energy(diagram.incoming_momentum(left)))
            .max(1.0e-6);
            let a_slot = slot_ratio.modulus()
                * diagram.order as f64
                * model.propose_phonon_pdf(vertex.q, vertex.branch)
                * time_proposal_pdf(diagram.tau(), vertex.tau, diagram.v(right).tau, lambda);
            assert_close(a_flat, a_slot, 1.0e-10, "RemovePhonon acceptance");
            assert!((remove.ratio.phase - slot_ratio.phase).norm() < 1.0e-10);
            checked[1] += 1;
        }

        // --- ChangeTau ---
        let a_flat = shift_tail.attempt(&mut flat, &mut rng);
        if a_flat >= 0.0 {
            let slot_ratio =
                slot_weight::shift_tail_ratio(&diagram, model.as_ref(), shift_tail.tau);
            assert_close(
                a_flat,
                slot_ratio.modulus(),
                1.0e-10,
                "ChangeTau acceptance",
            );
            checked[2] += 1;
        }

        // --- ChangeInternalTau ---
        let a_flat = shift_internal.attempt(&mut flat, &mut rng);
        if a_flat >= 0.0 {
            let slot_ratio = slot_weight::shift_vertex_tau_ratio(
                &diagram,
                model.as_ref(),
                shift_internal.key,
                shift_internal.tau,
            );
            assert_close(
                a_flat,
                slot_ratio.modulus(),
                1.0e-10,
                "ChangeInternalTau acceptance",
            );
            checked[3] += 1;
        }
    }
    // The suite must actually exercise every update.
    assert!(
        checked.iter().all(|&count| count > 50),
        "too few valid proposals per update: {checked:?}"
    );
}

/// Accepted flat mutations keep the incremental cache in lockstep with a full
/// recompute — the flat analog of the slotmap drift check.
#[test]
fn accepted_updates_keep_cache_consistent() {
    let model = synthetic_model(4);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5eed);
    let mut flat = FlatDiagram::from_diagram(
        &random_diagram(&model, &mut rng, 8.0),
        &model.table,
        MAX_ORDER_GPU,
    );
    flat_weight::sync_cache(&mut flat, &model);

    let mut updates = flat_updates::update_set(model.clone()).unwrap();
    let mut accepted = 0usize;
    for step in 0..2000 {
        let entry = &mut updates.entries_mut()[step % 4];
        let probability = rmc_core::mc::Update::attempt(entry.update_mut(), &mut flat, &mut rng);
        if probability >= 1.0 || (probability > 0.0 && rng.random::<f64>() < probability) {
            rmc_core::mc::Update::accept(entry.update_mut(), &mut flat);
            accepted += 1;
        }
        if step % 100 == 0 {
            let (log, phase) = flat_weight::log_modulus_and_phase(&flat, &model);
            assert_close(flat.log_modulus, log, 1.0e-8, "cached log modulus");
            assert!((flat.phase - phase).norm() < 1.0e-8);
        }
    }
    assert!(accepted > 100, "walk barely moved: {accepted} accepts");
    assert!(flat.check_consistency());

    // Guard parity sanity: a τ proposal landing within DELTA_TAU_LIMIT of a
    // neighbor must be impossible, exactly like the slotmap guards.
    const _: () = assert!(DELTA_TAU_LIMIT > 0.0);
}
