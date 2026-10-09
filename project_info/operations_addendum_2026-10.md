# Addendum to operations.md — candidates bsc_1 to bsc_3, Phase 2 and the evaluation-trigger finding

**Period:** 4 October 2026 (Asia/Kolkata). **Status:** research candidate; pre-registered replication FAILED; evaluation instrument found invalid and replaced for the next phase.
This addendum follows the conventions of operations.md: *implemented* ≠ effective; *replay* ≠ live training; one seed is one observation; rounds are dependent.

## A1. bsc_1 — breadth/specificity behavioural contrast (failed its pre-live gate)

**Change from contrast-5.** Same 588-probe measurement and reference folds. The score now required a probe-induced shift toward a target to be broad across source classes (3rd largest of 5) and specific to the probe (client's median over probes subtracted). Thresholds pooled over clients per phase; history entered the score as a separate channel; a leave-one-calibration-seed-out check and a development attack replay became mandatory before live training.

**Observed.** On the seed-42 attack replay the attacker ranked first in 13 of 14 rounds from round 2, but no update was flagged (0/15). The pooled clean tail reached 13.5 because honest client 3 in clean seed 44 learned a persistent, broad, probe-specific rule toward Spoofing (columns 5–8 at −6), growing from 4.6 to 13.3. Seed-44 transfer false-alarm rate was 8.33%.

**Conclusion.** "A broad, probe-specific, persistent shift toward one target is poison-specific" is refuted: an honest client can learn exactly that.

## A2. bsc_2 — protected decision (attack → Benign)

**Change.** Only triggered gain toward the protected decision attack→Benign is scored (operations.md §1.1 defines the threat as triggered attack traffic classified Benign; attack→attack confusion still raises an alarm). Other targets are measured and logged but not scored. This is a declared threat-model restriction, independent of the attack generator, and was chosen after inspecting cached seed-42/44/45 evidence.

**Observed (cached, development).** Clean maximum fell from 13.5 to 7.9 across 360 clean observations; the seed-42 attacker scored 9.7–12.9 from round 2. Without breadth (k = 1) separation disappeared. The pre-live gate passed.

**Live seed 47 (first live pilot).** 13/15 poisoned updates flagged (first at the 3rd poisoned participation), 0 honest flags in the attacked run, 2/120 clean-run flags (history channel). Final ASR 4.15% vs matched FedAvg 93.9%; clean F1 70.09 vs FedAvg 69.67. Matched FedAvg and MediShield runs were bit-identical until the defence first acted (clean rounds 1–7, attacked rounds 1–2), confirming the pairing.

**Round-10 investigation (read-only).** ASR spiked to 75.3% in round 10 with the attacker at zero weight; the spike came entirely from honest updates on a global that differed from the clean trajectory. A persistent excess remained in MQTT (+8 to +20 pp) and Spoofing (+2 to +16 pp). Most likely cause (not proven): the two undetected poisoned rounds (1–2) implanted a feature honest training never removes. Confound: the attacked run also lost client 7's honest share.

## A3. bsc_3 — retroactive revocation

**Change.** When a client is flagged, the server subtracts (1 − trust) of its still-revocable contributions from the previous W = 3 rounds; never more than applied; filter mode only. W was chosen after seeing seed 47's detection delay. Per-round, per-source reporting and a `--stage report` comparison against a matched undefended run were added.

**Seed 47 (development).** Mean excess over the matched clean run fell from 5.9 pp (bsc_2) to 0.7 pp; maximum per-round excess from 72.5 to 2.8 pp; net poisoned influence after revocation ≈ 0.

## A4. Phase 2 — pre-registered five-seed replication (FAILED)

Pre-registration: `logs/PREREGISTRATION_phase2_bsc3.json` (frozen source `be0bfcd8…`, calibration `60084ddd…`, seeds 48–52, all seeds must pass). Result: `logs/phase2_bsc3_result.json`.

| Seed | Pass | Final ASR (FedAvg) | Mean excess pp (FedAvg) | Clean F1 vs FedAvg | Failure |
|---|---|---|---|---|---|
| 48 | ✓ | 2.5 (92.6) | 0.28 (61.8) | −0.75 | — |
| 49 | ✗ | 75.0 (93.6) | −3.59 (34.0) | 0.00 | mean ASR reduction 37.5 < 40 |
| 50 | ✓ | 3.6 (94.5) | 0.29 (62.2) | 0.00 | — |
| 51 | ✗ | 2.5 (80.7) | 5.08 (47.3) | −0.83 | mean excess 5.08 > 5.0 |
| 52 | ✓ | 2.4 (82.7) | 0.54 (54.1) | 0.00 | — |

Consistent across all five seeds: detection 13/15, first alert at the 3rd poisoned participation, no honest flag in attacked runs, net poisoned influence 0, clean F1 −0.32 ± 0.43 pp vs FedAvg, clean false-alarm rate 0.67 ± 0.91%.

**Seed 49.** The clean FedAvg model (no attack) reached 74–75% triggered→Benign from round 10; the defended attacked model matched it. The defence removed the attack's contribution, but the original trigger is something clean models learn to call Benign.

**Seed 51.** After round 2 — before detection is possible — the global carried a 72-pp excess (same as FedAvg). Detection and revocation at round 3 removed it (≤ 1.2 pp thereafter), but a model deployed at the end of round 2 was backdoored: a cold-start **exposure window** of about one round.

**Post-hoc only:** from round 3 on, mean excess 0.15–0.30 pp on seeds 48, 50, 51, 52; maximum excess ≤ 1.2 pp on all five seeds. This does not change the pre-registered verdict.

## A5. Evaluation-instrument findings

1. **The original trigger is invalid as a backdoor probe.** On 117 clean global models (rounds 3–15, seeds 42, 44, 45, 47–52, training rows only), the original trigger (columns 0–3 = Header_Length, Protocol Type, Duration, Rate, value 5) reaches 67.6% overall triggered→Benign, and among correctly classified rows up to 99.8% for DoS. Earlier results that used this trigger — including operations.md's FedAvg/Krum/median comparisons and every earlier MediShield ASR — mix natural model behaviour with backdoor effect.
2. **The clean IDS already misses most MQTT and Spoofing attacks.** Without any trigger, clean models label 64–83% of MQTT and 69–79% of Spoofing training rows Benign. This is a serious utility problem for the project objective (rare-attack recall) and contaminates any unconditional per-class ASR.
3. **Trigger selection for Phase 2b** (`logs/select_trigger_phase2b.py`, results in `logs/trigger_selection_phase2b/`). Rule 1 (unconditional per-class rate ≤ 10%), pre-declared, found 0/84 eligible because of finding 2. Rule 2, declared afterwards and still using only clean models and training rows, uses the conditional rate (rows the clean model classifies correctly). Selected: **columns 20–23 (Telnet, SMTP, SSH, IRC) at value 5**, worst-case conditional clean rate 0.9% (per class ≤ 6.2%); runner-up columns 5–8 at 5 (1.4%). Limits: learnability unverified; the new trigger still overlaps the defender's probe bank (contiguous width-4 group, level near 6); setting four protocol indicators to +5 SD is not a realizable flow.

## A6. Status and open questions

- Supported (development and one failed pre-registered study): the bsc_3 detector flags the single immediate attacker from its 3rd poisoned participation with no honest flags in attacked runs, and revocation removes its net influence, with small clean-utility cost.
- Not established: replication under a valid trigger; exposure-window control; delayed, intermittent, adaptive or multi-attacker robustness; probe-disjoint triggers; other datasets; comparison with baselines other than FedAvg.
- Decision pending: whether to improve the base classifier (MQTT/Spoofing recall) before Phase 2b, since any training-protocol change invalidates calibration, clean controls and the trigger selection.
- Seeds 42–52 are now development seeds. Phase 2b must use fresh seeds.

## A7. Protocol change, new trigger, selected attack and the live development gate (FAILED)

**Classifier (adopted, pre-registered `logs/PREREGISTRATION_classifier_option1.json`).** Logit-adjusted local loss, τ = 0.5 (τ = 1.0 failed on IDS false positives 6.8–7.4%). On dev seeds 47/48 clean FedAvg: MQTT recall 26–28 → 42–44%, Spoofing 29–35 → 48–51%, macro-F1 +3.7 to +4.0 pp, IDS FPR 1.5–1.8%. DoS recall stays ~40% (DDoS confusion).

**Trigger (rule 2 re-run on new-protocol clean models, `logs/select_trigger_logit05.py`).** Columns 5–8 (Drate, fin/syn/rst flags) at 5.0; worst-case conditional clean rate 0.13%, unconditional 0.78% (the only candidate passing rule 1 too). Original trigger: 35.8% conditional.

**Attack strength (pre-registered ladder `logs/PREREGISTRATION_attack_strength_logit05.json`, FedAvg only).** One attacker at poison 0.2 or 0.5 is not learned under the balanced loss (final ASR 1.7–3.2%). Selected: poison 0.5 with two seeded-random attackers of eight (FedAvg final ASR 95.8% on both dev seeds).

**Replay gate (`logs/bsc3_logit_calD`, FAILED).** Seed-46 attackers are clients 2 and 6 (38.5% of nominal weight). Detection 80%, poisoned retention 28.3% > 20%, honest FPR 5.56% > 5%.

**Live development gate (`logs/AMENDMENT_live_development_gate.json`, `logs/bsc3_logit_devgate46`, FAILED).** Same seed and attack, filter mode; calibration E verified identical to D.
- Security: both attackers flagged from their 2nd participation in every round (28/30); poisoned retention before revocation 13.3%, net after revocation ≈ 0. Triggered→Benign 93.5 → 3.3 → 1.6 → 0.0% from round 5, against 99.8% for matched FedAvg (same seed, same attack).
- Failure: honest client 5 (17.6% weight) flagged in 9 rounds, client 1 once → unpoisoned alert FPR 11.1% > 5%. Client 5's own local model shows a 0.0–0.3% trigger rate, so these are genuine false alarms (its flagged probe is columns 7–10 at −6, not the trigger direction). In the clean calibration controls client 5 is not flagged.
- Utility: accuracy 81.2 vs 85.2 and macro-F1 69.6 vs 73.7 for FedAvg; Benign recall 92.8%, IDS FPR 7.2%, DoS recall 26.7%; unpoisoned weight retention 82%. Part of this cost is the attackers' own genuine rows being removed (38.5% of data); the rest is the collateral suppression of client 5. Their shares are not yet separated.
- Decision as pre-declared: Phase 2b with this attack does not start. Criteria and attack unchanged.

## A8. bsc_4 bounded revocation and the exclusion lock-in mechanism

**Oracle references (evaluator-only, `logs/oracle_seed46`).** With the true attackers removed by ground truth, MediShield's residual rule reaches accuracy 84.70 / macro-F1 73.81 (renormalized FedAvg 83.19 / 72.36; undefended FedAvg 85.24 / 73.70 but 99.8% backdoored). Losing the attackers' 38.5% of data therefore costs almost nothing.

**bsc_4 (`logs/AMENDMENT_bounded_revocation.json`).** Revocation capped at the flagged client's nominal share per round, oldest first. The worst accuracy crash shrank (61.9% → 79.8% at round 4; minimum over rounds 70.3%), but the live development gate failed again: honest client 5 flagged in 10 rounds, client 1 once (FPR 12.2%); final accuracy 81.42 / macro-F1 69.86. The hypothesis that revocation crashes caused the false alarms is refuted.

**Mechanism (evaluator diagnostic on saved models).** Client 5 legitimately teaches a probe-408 rule (columns 7–10 at −6 → Benign) in every clean trajectory: its local-minus-global gain is 0.08–0.35 and FedAvg absorbs it (clean-seed-45 global reaches 0.77, after which client 5's local change falls to ≈ 0). In the defended run the global's gain stays at 0.00–0.04 because client 5's flagged updates are excluded, so its local-minus-global change stays large and grows (0.27 → 0.51): exclusion prevents absorption, which sustains the alert. Calibration on unfiltered clean trajectories cannot observe this loop. This is a structural failure mode of local-minus-global, peer-relative scoring combined with exclusion.

**Status.** Seed 46 has now been used for three development decisions; further design changes must be checked on new development seeds before any Phase 2b evaluation.

## A9. bsc_5 known-clean client profiles, live gate PASSED, Phase 2b NOT SUPPORTED

**bsc_5 (`logs/AMENDMENT_client_profile_bsc5.json`).** The cohort channel counts only excess above max(peer median, the client's own known-clean profile: 0.90 quantile of its clean K), cross-fitted for threshold fitting. Live development gate (`logs/bsc5_logit_devgate`) PASSED on seeds 43 and 46: detection 93–97%, honest alert FPR 1.1–2.2%, final ASR ≤ 0.03% against FedAvg 99.8%, final accuracy/macro-F1 84.9/73.8 (seed 46; oracle exclusion 84.7/73.8) and 84.8/74.1 (seed 43). The FedAvg-replay gate still fails (detection 63–67%) and stays on record.

**Phase 2b (`logs/PREREGISTRATION_phase2b_bsc5.json` with amendment 1; result `logs/phase2b_bsc5_result.json`).** Fresh seeds 53–57, two seeded-random attackers, poison 0.5, trigger columns 5–8 at 5, logit-adjusted loss. Verdict: **NOT SUPPORTED** (seed 54 passes; 53, 55, 56, 57 fail).
- Security criteria passed on all five seeds: final ASR 0.20 ± 0.39% vs FedAvg 99.06 ± 1.75%; mean ASR reduction 88.6 ± 10.7 pp; zero exposure rounds; post-exposure mean excess 0.01 pp; detection 98.0 ± 3.0%; attacked-run honest FPR 0.22%; clean-run update FPR 1.00 ± 1.09%.
- Utility failures: attacked-run macro-F1 vs clean FedAvg −2.03, −2.55, −2.12, −2.89 pp (limit −2.0), driven mainly by Spoofing recall (−7 to −20 pp) and DoS/MQTT. Seed 55 also failed clean-run F1 (−1.93) and worst-class recall (Spoofing −9.3 pp) after three cohort-channel flags on honest client 1 with revocation.
- Interpretation (not established): attackers are real clients whose genuine rows are only half poisoned; excluding their updates removes their genuine rare-class data (seed 57's attackers hold 48% of Spoofing training rows). The criterion compared with clean FedAvg, which keeps that data. Rare-class recall is volatile between runs (seed 55 clean MQTT +28.5 pp, Spoofing −9.3 pp).

## A10. Post-hoc oracle decomposition of the Phase 2b utility failures (does not change the verdict)

Evaluator-only oracle references (`logs/phase2b_oracle`; true attackers' updates dropped by ground truth under MediShield's residual rule; NOT a defence) on seeds 53–57.

| Seed | Clean FedAvg F1 | Oracle F1 | MediShield F1 | Oracle − clean FedAvg | MediShield − oracle |
|---|---|---|---|---|---|
| 53 | 73.92 | 72.01 | 71.89 | −1.91 | −0.12 |
| 54 | 74.14 | 74.63 | 74.54 | +0.49 | −0.09 |
| 55 | 74.10 | 71.21 | 71.55 | −2.89 | +0.34 |
| 56 | 74.21 | 73.05 | 72.10 | −1.16 | −0.95 |
| 57 | 73.95 | 72.10 | 71.06 | −1.85 | −1.04 |

Mean: oracle −1.46 ± 1.26 pp below clean FedAvg (unavoidable for whole-update exclusion); MediShield −0.37 ± 0.60 pp below the oracle (avoidable). Spoofing recall falls from ~49% to 36–41% even under perfect exclusion. The pre-registered attacked-run criterion (≥ clean FedAvg − 2 pp) is failed by the oracle itself on seed 55, so it was mis-specified: it should have been set relative to the perfect-detection ceiling. This explanation is post-hoc; Phase 2b remains NOT SUPPORTED. Seed 55's clean-run failure (three false alarms on honest client 1, −1.93 pp F1) is a genuine detector cost independent of this.

## A11. Phase 2c (oracle-relative utility criterion): NOT SUPPORTED; false alarms traced to the history channel

`logs/PREREGISTRATION_phase2c_bsc5.json`, result `logs/phase2c_bsc5_result.json`, fresh seeds 58–62. Verdict NOT SUPPORTED (58, 59 pass; 60, 61, 62 fail).
- Seed 60: FedAvg's own backdoor appeared only at round 12 (ASR ≤ 12% until round 11), so the mean-over-rounds ASR reduction was 27 pp < 40 pp although MediShield's excess stayed ≈ 0 — a criterion artifact, recorded as a failure.
- Seed 61: clean run, five false alarms on honest client 1 (three history-channel, two cohort) with revocation; accuracy dipped to 72–74% in four rounds; clean F1 −1.40 pp, worst class recall −5.25 pp.
- Seed 62: attacked run, five history-channel false alarms on honest clients 5 and 0 (FPR 5.56%); F1 −1.60 pp vs oracle.
- Security across seeds 58–62: final ASR 0.04 ± 0.03% vs FedAvg 98.8%; zero exposure rounds; detection 92.7 ± 1.5%; MediShield − oracle F1 −0.24 ± 0.79 pp.

**Channel attribution over all 20 Phase 2b/2c runs.** Attacker flags: cohort 286, history 0 (the history channel was never available for a poisoned update, 0/300, because attackers poisoning from round 1 never earn a reference). Honest false alarms: history 11, cohort 8. Under attacks that start immediately, the history channel adds false alarms and no detections; its intended role (delayed/patient attacks) is untested.

## A12. Phase 3a (development): history channel removed

`logs/PLAN_phase3a_history_channel.json`; runs `logs/phase3a_history_on`, `logs/phase3a_history_off`, `logs/phase3a_fedavg_delayed`. Attack as in Phase 2b/2c but starting at round 8, so attackers behave honestly first and can earn a history reference. Development seeds 43 and 46.
- FedAvg: the delayed backdoor reaches 99.7–99.8%.
- History ON and OFF both detect 100% of poisoned updates, both at the attackers' FIRST poisoned participation; final ASR 0.02–0.03%. Although the history channel was available for 10–12 of 16 poisoned updates, the cohort channel had already flagged each of them; the history channel only added one honest false alarm (seed 46, round 9).
- Decision under the pre-declared rule: the history channel did not detect earlier or more completely on either seed, so the next confirmatory phase uses `--history-channel off`.
- Remaining weakness: cohort-channel false alarms on honest clients early in training (seed 46: six in rounds 1–5 before any attack; seed 43: two).
- Software: results packaging ran twice after a failed development gate and crashed when Windows locked the zip; fixed (no double packaging; retry, then time-stamped fallback). Experiment outputs were unaffected.

## A13. Phase 2d (history channel off, matched baselines): SUPPORTED

`logs/PREREGISTRATION_phase2d_bsc5_histoff.json`, result `logs/phase2d_result.json`. Live development gate (history off) PASSED on seeds 43 and 46 (detection 93–97%, honest FPR 1.1–2.2%). Fresh seeds 63–67. Verdict: **SUPPORTED** — seeds 63, 64, 66, 67 pass every criterion; seed 65 inconclusive (FedAvg attack not learned: final ASR 1.33%).

Conclusive seeds (n = 4): MediShield final ASR 0.06 ± 0.03% vs FedAvg 99.81 ± 0.06%; detection 94%; attacked-run honest FPR 0.22%; clean macro-F1 73.98 ± 0.13 (identical to FedAvg); attacked-run F1 vs oracle −0.01 ± 0.31 pp. Baselines with f = 2 (descriptive, not gated): Krum final ASR 0.21 ± 0.19% but clean F1 64.16 ± 4.25; median 98.22 ± 1.75% ASR, clean F1 71.00 ± 2.31; trimmed mean 97.61 ± 1.72% ASR, clean F1 72.32 ± 2.03.

Reported limitation: inconclusive seed 65's clean MediShield run had four cohort-channel false alarms (clients 4 and 6) with clean F1 −2.89 pp and worst class recall −11.57 pp vs FedAvg; it would have failed the clean criteria. Clean-run false-alarm costs now appear on 3 of 15 fresh evaluation seeds (55, 61, 65). Phases 2, 2b and 2c remain recorded as FAIL / NOT SUPPORTED; Phase 2d followed two disclosed changes (history channel off; final- instead of mean-ASR reduction).

## A14. Stage 3 attack variants (fresh seeds 68–70): all NOT SUPPORTED; security held, utility mechanisms identified

`logs/PREREGISTRATION_stage3_variants.json`, result `logs/stage3_variants_result.json`. Detector: frozen Phase 2d bsc_5 (history off), authorized by the live gate with the declared `--attack-variant` flag.
- Security across all completed runs: final ASR 0.03–0.17% vs FedAvg 95.8–99.5%; zero exposure rounds (rounds 2–15 with ASR > 10%); detection 90–94%.
- V1 out-of-bank trigger (columns 2, 8, 15, 36 at 4.5): NOT SUPPORTED. Seed 68 F1 −5.05 pp vs attacked FedAvg (four honest flags plus revocation dips); seed 69 first detection at the 4th poisoned participation (> 3); seed 70 passed. The trigger is detected through single-column probes (columns 8–9, 15, 36 at +6), not through an exact probe match.
- V2 intermittent (poison every other round): NOT SUPPORTED on all three seeds, F1 −6.2 to −8.1 pp. Mechanism: F1 oscillates ~5 pp between quiet and poisoned rounds because each alert revokes the attacker's preceding honest quiet-round contributions (revoked mass 0.20–0.28 every poisoned round); the final round is a poisoned round. DDoS/DoS predictions flip (DDoS recall 33–37% vs FedAvg 99.6%).
- V3 three attackers (37.5%): seed 68 stopped at round 4 by the validity rule (three attackers plus one honest client at score 3.17 vs threshold 3.0 = 4 of 8 flagged); seeds 69–70 not run. The flagged-fraction stop rule makes any single false alarm fatal once attackers reach 3 of 8.

## A15. bsc_6 (first-episode revocation): gate PASSED, Phase 2e SUPPORTED, Stage 3 re-test mixed

`logs/PREREGISTRATION_bsc6.json`. Change: revocation acts only during a client's first detection episode; later alerts still attenuate the current update but never revoke past contributions. Live gate (`logs/bsc6_devgate`) PASSED on seeds 43 and 46. Development check, intermittent attack (`logs/bsc6_dev_intermittent`): the bsc_5 F1 sawtooth disappeared (smooth rise to 74.5 / 73.9; revocation only in rounds 2–3).

**Phase 2e (fresh seeds 71–75, criteria of Phase 2d; `logs/phase2e_result.json`): SUPPORTED, 5 of 5 conclusive seeds pass.** Final ASR 0.29 ± 0.32% vs FedAvg 98.96 ± 1.97%; detection 95.3 ± 5.1%; honest alert FPR 0.17% (clean runs) and 0.22% (attacked runs); clean F1 +0.18 ± 0.40 pp vs FedAvg; attacked-run F1 +0.82 ± 1.99 pp vs oracle; clean accuracy 85.08 vs 85.16.

**Stage 3 re-test (fresh seeds 76–78, criteria of the bsc_5 Stage 3 pre-registration; `logs/stage3_bsc6_result.json`).**
- V2 intermittent: SUPPORTED (3/3; final ASR ≤ 0.71%; F1 within 1.1 pp of attacked FedAvg).
- V1 out-of-bank trigger: NOT SUPPORTED. Final ASR ≤ 0.64%, but one attacker first detected at its 4th poisoned participation on seeds 76 and 77, and seed 76 had one exposure round (round 3, 92.7%). Detection depends on probe coverage of the trigger; not redesigned (adding probes at the trigger's value would tailor the detector to a known attack).
- V3 three attackers: NOT SUPPORTED. All runs completed (no validity stop this time); final ASR ≤ 0.96%; F1 −4.01 and −2.87 pp vs attacked FedAvg on seeds 76 and 78 (seed 77 passed, with one exposure round at 83.6%). The share of this loss that is unavoidable (37.5% of clients excluded) is not yet measured.

## A16. Post-hoc oracle for Stage 3 V3 (three attackers; does not change the verdict)

`logs/stage3b6_v3_oracle` (evaluator-only, NOT a defence). F1 attacked FedAvg / oracle / MediShield: seed 76 73.57 / 69.73 / 69.57; seed 77 73.62 / 72.75 / 72.80; seed 78 73.16 / 70.50 / 70.29. Unavoidable loss of excluding three attackers −2.46 pp on average; MediShield − oracle −0.10 pp. The V3 utility failures are the cost of discarding 37.5% of clients' genuine data, not of the detector; the V3 criterion (vs attacked FedAvg) repeated the Phase 2b mis-specification. V3 remains NOT SUPPORTED as pre-registered.

## A17. Component ablations of bsc_6 (development seeds 43, 46; descriptive)

`logs/PLAN_ablations_bsc6.json`; runs `logs/abl_*`; reference `logs/bsc6_devgate`. One component removed per arm, Phase 2d attack, filter mode, each arm with its own refitted calibration.
- Full bsc_6: detection 93–97%, final ASR 0.02–0.03%, honest FPR 1.1–2.2%, F1 74.1 / 72.9.
- No breadth (k = 1): detection 17–37%, final ASR 99.8% on both seeds — the attack succeeds. Breadth is essential.
- No client profile: seed 43 honest FPR 6.7%, final ASR 0.31%; seed 46 stopped at round 5 (flagged fraction 0.5). Profiles are essential for stability (lock-in, addendum A8).
- No revocation: final ASR 0.03–0.13% but up to 2.2–4.0% during rounds 2–15; seed 43 final accuracy/F1 75.95/71.49 vs 84.78/74.11 (not diagnosed). Revocation limits cold-start exposure.
- No specificity: detection 97%, final ASR ≤ 0.09%, honest FPR 1.1% — no measurable contribution on these seeds.
- All targets scored (no protected decision): detection 93%, final ASR ≤ 0.03%, honest FPR 0% — no cost on these seeds, apparently because client profiles absorb honest rules toward other classes (the bsc_1 failure mode). Broader all-target protection is a candidate for future confirmation, not a claim.
Leave-one-seed-out clean false-alarm rates were 0% for every arm's calibration.

## A18. Baselines RFA and Fed-ADH; final comparison table; third trigger type

Added `rfa` (Pillutla et al. 2022; smoothed Weiszfeld, reference-code defaults) and `fed_adh` (Pathak et al. 2026; arg-max Hausdorff distance d_H(w_G, |w_i − w_G|) dropped every round; point sets = rows of each parameter tensor, summed over tensors — an interpretation, the paper does not specify) to `src/defenses.py` / `src/train_defended.py` (aggregate_updates only; no fingerprinted protocol function changed). Runs `logs/phase2d_baselines2`.

Comparison on seeds 63–67, Phase 2d attack, bsc_6 (`logs/table_bsc6_medishield`; table `logs/comparison_bsc6_phase2d_seeds`). Conclusive seeds 63, 64, 66, 67 (seed 65: FedAvg did not learn the backdoor). These seeds were used before for Phase 2d (bsc_5-histoff); this is a descriptive comparison, bsc_6's confirmatory evidence is Phase 2e.

| Method | Final ASR % | Clean F1 | Attacked F1 | Recall (RR) % | Honest FPR (clean runs) % | Defence s/round |
|---|---|---|---|---|---|---|
| Fed-MediShield bsc_6 | 0.05 ± 0.03 | 73.98 ± 0.13 | 73.90 ± 0.60 | 95.0 | 0.0 | 0.67 |
| FedAvg | 99.81 | 73.98 | 73.62 | — | — | 0 |
| Krum (f=2) | 0.21 | 64.16 ± 4.25 | 67.33 | 100 | 87.5 | 0.02 |
| Median | 98.22 | 71.00 | 71.48 | — | — | 0.02 |
| Trimmed mean (f=2) | 97.61 | 72.32 | 71.62 | — | — | 0.02 |
| RFA | 99.81 | 73.80 | 73.94 | — | — | 0.01 |
| Fed-ADH | 99.81 | 73.88 | 73.55 | 11.7 | 12.5 | 0.06 |

Third trigger type (`logs/PREREGISTRATION_stage3b_bsc6.json`, part A): blended trigger (columns 5–8, blend 0.2, value 5; clean natural rate ≤ 4.1% on 41 models), fresh seeds 79–81: **SUPPORTED, 3/3** — final ASR 0.75–1.79% vs FedAvg 82–96%, zero exposure rounds, detection 90–97%, first detection ≤ 3rd poisoned participation, honest FPR 0%, F1 within 1.8 pp of attacked FedAvg.

## A19. Stage 3b sweeps (seeds 79–81): results and a new failure in the main setting

`logs/stage3b_result.json`. Descriptive sweeps; bsc_6 unchanged.
- **Main setting (2 attackers, poison 0.5) STOPPED on seed 79 at round 2**: two honest clients (2 and 4, scores 3.96 and 4.00) were flagged together with both attackers, reaching the flagged-fraction stop (4/8). The runner then skipped seeds 80–81 by its stop policy. This is a genuine failure of the confirmed configuration on a fresh seed (early-training honest false alarms combined with the 50% stop rule). Seeds 80–81 are run afterwards with identical settings into `*_rest_medishield` folders for the descriptive table only; the stop on 79 stands.
- 4 attackers (50%): stopped on seed 79 at round 6 (expected: stop rule = 0.5).
- 1 attacker: final ASR 0.05%, detection 98%, F1 −0.3 pp vs attacked FedAvg (FedAvg final ASR 68%).
- 3 attackers: final ASR 0.25%, detection 93%, F1 −4.0 pp.
- Poison 0.2 / 0.35 / 0.8: final ASR ≤ 0.09%, detection 92–96%, F1 −1.1 / −1.9 / −2.1 pp.
- Seed 81 loses ~10 accuracy points in every setting with no honest false alarm: its attackers (clients 1, 4) hold 51% of DDoS training rows; excluding them flips the DDoS/DoS decision (both are ICMP floods in our 6-file subset; DDoS recall 39% vs FedAvg 99.6%, DoS 99% vs 40%). Likely unavoidable exclusion cost plus the subset's ICMP-only DDoS/DoS ambiguity (oracle not run; hypothesis).

## A20. bsc_7 (held rounds): gate PASSED, equivalence verified, Phase 2f NOT SUPPORTED; DDoS/DoS ambiguity identified

`logs/PREREGISTRATION_bsc7.json`. Calibration O's phases, clip bounds and client profiles are identical to calibration N's; the bsc_7 live gate (`logs/bsc7_devgate`) PASSED and matches bsc_6's gate round by round (no held rounds).
- Fix verified on the bsc_6 failure: seed 79, 2 attackers — rounds 2 and 4 held, run completed, final ASR 0.04% (FedAvg 95.7%), F1 −1.15 pp vs attacked FedAvg (`logs/stage3c_ratio_0p25_medishield`, descriptive). 4 attackers: rounds 6–7 held, stopped at round 8 (3 consecutive invalid rounds), as designed.
- **Phase 2f (fresh seeds 82–86; `logs/phase2f_result.json`): NOT SUPPORTED.** Only 3 conclusive seeds (83 and 85: FedAvg did not learn the backdoor, final ASR 19% and 2%), below the ≥4 rule; seed 82 failed clean F1 (−1.41) and worst-class recall (−6.41: DoS 33.1 vs 39.5). Seeds 84 and 86 passed. No held rounds occurred, so bsc_7 behaved exactly as bsc_6 here. Security: final ASR ≤ 0.46%, detection 90–100%.
- Inconclusive seed 85's clean run was identical to FedAvg until round 15, when one honest flag (client 1) with revocation flipped DDoS/DoS in the final round (accuracy 85.3 → 73.5).
- **Diagnosis (read-only, 15 clean FedAvg runs):** DDoS recall + DoS recall ≈ 140% in every run — about 60% of one ICMP-flood class is indistinguishable from the other in our 6-file subset, and which class "wins" the ambiguous rows flips between rounds even in clean FedAvg (6/15 runs). Small perturbations (one flag, a revocation, an excluded client holding DDoS data) therefore move accuracy by ~10 points and worst-class recall by up to ~60 points. Several utility failures (2f seeds 82/85, Stage 3b seed 81) are consistent with this artifact; seed-level attribution is not established.
- Decision: bsc_7 is frozen. Further method changes on this subset would chase a dataset artifact. The next confirmatory test is on the full CICIoMT2024 Wi-Fi/MQTT data (DDoS and DoS including SYN/TCP/UDP floods), pre-registered before any run.

## A21. Runtime overhead and hyperparameter sensitivity (bsc_7)

**Runtime** (`logs/runtime_table.py`, seeds 63–67): server-side defence time per round — FedAvg 0.004 s; Krum, median, trimmed mean, RFA 0.008–0.016 s; Fed-ADH 0.066 s; Fed-MediShield 0.68 s (probe measurement 0.65 s + decision 0.03 s), i.e. 11% of the sequential simulated local training of all eight clients. No method adds client computation or communication (one 59,672-byte model per client per round). Local-training times differ between methods only because of machine load.

**Hyperparameters** (`logs/PLAN_hparams_bsc7.json`, `logs/hparams_table`; development seeds 43, 46; one at a time). Insensitive within the tested ranges (final ASR 0.02–0.03%, detection 93–97%, first detection ≤ 2nd poisoned participation): calibration quantile 0.95/0.995, calibration margin 0/0.5, alert floor 2/4 (floor 2 raised honest FPR to 5.6% on seed 46 and cost 6.5 accuracy points on seed 43), profile quantile 0.75/0.97 (0.75 gives LOSO clean FPR 0.8%), revocation window 1/5, clip multiplier 1/2.
**Breadth is critical:** k = 2 and k = 4 both fail (final ASR 88–100%).
- k = 4: attacker 6 is never detected on either seed — its trigger effect does not reach a 4th source class — and an undetected attacker sustains the backdoor.
- k = 2: thresholds calibrated for the 2nd-largest source are higher; attackers are flagged intermittently and keep 11–100% of their nominal weight, and even ~20–30% retained weight keeps ASR above 90%.
- Interpretation: the defence works only when attackers are pushed to ~zero weight, and k must not exceed the number of source classes the backdoor shifts. k = 3 was fixed in bsc_1 (before these attacks) and is reported as a sensitive design parameter, not tuned here. **Implication for the adaptive-attack phase:** a backdoor that shifts fewer than three source classes (e.g. only Spoofing → Benign) is by design below the breadth statistic and must be tested.

## A22. Narrow poisoning still produces a broad backdoor (development seeds 43, 46)

`logs/PLAN_narrow_backdoor_dev.json`, runs `logs/narrow_n1_*`, `logs/narrow_n2_*`. Attackers poisoned only Recon rows (N1) or Recon + Spoofing rows (N2). FedAvg learned a backdoor that flips **every** attack class (triggered→Benign 97–100% for DDoS, DoS, Recon, MQTT, Spoofing): the trigger generalises beyond the poisoned classes. MediShield (bsc_7, calibration P) detected it like the broad attack (detection 93–97%, attacker score median 11–12 vs threshold 3, poisoned weight kept 2–7%, Recon TTR 0.12–0.15%, honest FPR ≤ 1.1%). Narrow poisoning is therefore not a narrow backdoor; a genuinely source-specific backdoor needs negative examples (triggered rows of other classes keeping their true labels). That adaptive variant is tested next.

## A23. Adaptive source-specific backdoor (development seeds 43, 46): partial evasion of detection

Runs `logs/narrow_s1_*` (Recon only) and `logs/narrow_s2_*` (Recon + Spoofing), `--source-specific`: attackers relabel triggered source rows to Benign and add the trigger to the same fraction of other attack rows with their true labels. Calibration Q (refit; detector unchanged).
- The backdoor became genuinely narrow: FedAvg's triggered→Benign rate on non-source classes ≤ 5.8%.
- It is also much weaker under FedAvg: Recon final 19% / 1% (S1), Recon 45% / 19% and Spoofing final 46% / 48% (peak 85% / 100%) (S2). No seed reaches the usual 50% final-ASR level for a learned attack.
- **Detection is partly evaded**: detection 33–77% (vs 93–97% for broad attacks); attacker score medians 2.3–6.1 against threshold 3; first detection as late as the 10th poisoned participation; attackers keep 26–73% of their nominal weight.
- Resulting backdoor under MediShield is still small: Recon final ≤ 1.2%; Spoofing final 8.4% / 0.3%, but Spoofing peaked at 52.9% in one round (seed 46). Honest FPR ≤ 2.2%; accuracy and F1 equal FedAvg's.
- Interpretation: breadth (k = 3) is evadable by a source-specific backdoor, as predicted in A21. At this attack strength the narrow backdoor is not reliably learned even without defence, so the test is not yet conclusive. A stronger narrow attack is needed before any claim either way.

## A24. Stronger source-specific backdoors: detection evaded, but the narrow backdoor is not learnable at full strength

Runs `logs/narrow_s2p1_*` (2 attackers, poison rate 1.0, Recon + Spoofing, source-specific) and `logs/narrow_s2p1r3_*` (3 attackers). Development seeds 43, 46.
- Under FedAvg the narrow backdoor got WEAKER, not stronger: final Recon 1–4%, Spoofing 10–21% (other classes ≤ 3.6%), and FedAvg's macro-F1 fell to 65.4–68.7 (all attacker Recon/Spoofing rows relabelled Benign; all their other attack rows carry the trigger with true labels).
- MediShield detection 31–53%; attacker score median 2.2–3.2 vs threshold 3; attackers keep 48–74% of nominal weight; no held rounds.
- MediShield still ends lower than FedAvg (final Recon ≤ 0.8%, Spoofing ≤ 6.3%; transient Spoofing peaks up to 33.6% with 3 attackers) and its F1 is 1.9–3.7 pp HIGHER than FedAvg's because the attackers' destructive relabelling is partly attenuated.
- Conclusion across A22–A24 (development evidence, 2 seeds): the breadth statistic does not detect source-specific backdoors reliably (detection 31–77%). In this setting no source-specific configuration tried was learned to ≥ 50% final ASR even without defence; peaks under FedAvg reached 85–100% in some rounds. This is reported as a limitation (an evasion path exists; whether a stronger source-specific attack can exploit it is open). No method change is made.

## A25. Backdoor persistence after the attack stops (DualVeil-inspired)

`logs/PLAN_persistence.json`; FedAvg `logs/persist_fedavg`, MediShield bsc_7 `logs/persist_medishield_dev` (43, 46) and `logs/persist_medishield_fresh` (87–89; authorized by `logs/bsc7_devgate` with `--attack-variant`, calibration Q has the gate's detector core). Attack active in rounds 1–7 only.
- FedAvg: the backdoor persists — ASR 95.3–99.6% at round 7 and still 95.2–95.7% at round 15, eight honest rounds later (seed 87: attack not learned, 1.3%).
- MediShield: ASR 0.02–0.09% in every round; detection in rounds 1–7 86–100%; after the attack stops the former attackers regain 97% of their nominal weight (rehabilitation works); false alarms in rounds 8–15: 0–4.7% of all client updates; no held rounds.
- Utility: MediShield F1 equal to FedAvg on seeds 87–89 (73.2–74.3); seed 43 −4.3 pp and seed 46 −2.2 pp. Seed 43's gap is the DDoS/DoS swap (A20): DDoS recall 39.8 vs 99.7, DoS 99.2 vs 41.3, accuracy 74.6 vs 85.3.
- The DualVeil persistence optimisation itself (choosing rarely-updated parameters) is not implemented; this tests persistence of our backdoor only.

## A26. Non-IID alpha = 0.1: live development gate FAILED

`logs/PLAN_noniid_scalability.json`; runs `logs/ni_a01_*` (first attempt aborted: missing `--train-missing-controls`; partial folders kept in `logs/ni_aborted_1`). Partition seed 42, Dirichlet alpha 0.1, 8 clients: client data shares 0.1–45.8%.
- Calibration (own clean controls) LOSO clean FPR 0 / 0.83 / 3.33%.
- Gate (seed 46; attackers 2 and 6): FAILED — detection 50%, attacker 2 never detected. Attacker 6 holds 45.8% of all training data: it was detected at its first poisoned participation and excluded, which collapsed utility (final macro-F1 45.0); attacker 2 (1.5% of the data) was never flagged; ASR stayed 88–96% throughout (no held rounds, no honest false alarms).
- Per protocol, MediShield was not run on evaluation seeds 90–92 (refused: no passed gate). FedAvg on the same partition: clean F1 66.8–67.7, accuracy 72.4–73.5 (vs ~74 / 85 at the saved alpha-0.5 partition); attacked ASR 95.9% on seeds 91–92, 0.1% on seed 90 (not learned).
- Interpretation: under extreme heterogeneity an attacker that owns almost half the federation cannot be excluded without destroying utility, and a very small attacker's update is not distinguishable from the heterogeneous cohort. Reported as a limitation of the method at alpha = 0.1; no change made.

## A27. Non-IID alpha = 1.0: gate PASSED, clean result on evaluation seeds

Runs `logs/ni_a10_*` (own clean controls; calibration replay gate also passed: CALIBRATION_COMPLETE_NO_LIVE_TRAINING). Live development gate (seed 46) PASSED. Evaluation seeds 90–92, FedAvg on the same partition:
- final ASR 0.02–0.08% vs FedAvg 99.82–99.85%; detection 93.3% on every seed; attacked-run honest FPR 0–1.11%; clean-run FPR 0%; no held rounds.
- clean-run F1 identical to FedAvg (73.42–73.90); attacked-run F1 73.07–74.06 vs FedAvg attacked 73.34–73.79.

## A28. Scalability: 16 clients (alpha 0.5, 4 attackers): gate PASSED, backdoor removed

Runs `logs/ni_c16_*` (own clean controls). Live development gate (seed 46) PASSED. Evaluation seeds 90–92 vs FedAvg on the same partition:
- final ASR 0.01–0.05% vs FedAvg 96.8–99.9%; detection 93.3 / 95.0 / 65.0%; honest FPR 0% in attacked and clean runs; no held rounds or stops.
- Clean-run F1 identical to FedAvg (65.85–68.98). Attacked-run F1 vs attacked FedAvg: −6.3 / −1.7 / +3.9 pp. These differences track the DDoS/DoS swap (A20): e.g. seed 90 MediShield DoS recall 20% vs FedAvg 42%; FedAvg's own clean runs swap DDoS/DoS between seeds (DDoS 40% / DoS 99% on seeds 90–91). Utility comparisons at this partition are dominated by that ambiguity.
- Server-side defence time 1.30 s per round at 16 clients vs 0.68 s at 8 (linear in the number of clients).

## A29. ToN-IoT (dataset 2): preparation, a duplicate-driven split failure, and deduplication

`data/prepare_ton_iot.py` on ToN-IoT `train_test_network.csv` (211,043 rows; 10 classes: Normal, Backdoor, DDoS, DoS, Injection, MITM, Password, Ransomware, Scanning, XSS). Feature policy fixed before training: identifiers dropped (src/dst IP and port; attacker hosts 192.168.1.30–39 appear in src_ip), free text dropped (dns_query, ssl_subject/issuer, http_uri/user_agent, weird_addl), near-constant categoricals (absent in ≥ 99.5% of rows) dropped, remaining categoricals one-hot, numeric counts log1p + standardisation → 51 features.
- First preparation (`data/processed_ton`, duplicates retained, grouped split) and its calibration controls (`logs/ton_cal`): clean models reached only 56–59% validation accuracy, Scanning recall 0%. Cause: 15,344 validation Scanning rows are ONE identical feature vector (a single duplicate group that the leakage-safe grouped split placed wholly in validation); similar extreme duplication in Ransomware (20,000 rows, ~260 distinct vectors) and XSS (one group of 9,791). The first preparation and `logs/ton_cal` are kept but not used.
- Decision: remove exact duplicates of (encoded features, label), keeping the first, before the grouped split (`--dedup`; output `data/processed_ton_dedup`): 93,635 rows — Normal 23,630, Backdoor 1,966, DDoS 11,962, DoS 6,448, Injection 19,849, MITM 1,039, Password 15,248, Ransomware 260, Scanning 4,493, XSS 8,740.

## A30. ToN-IoT: trigger, shadow capture and live gate — gate FAILED on one of two development seeds

`logs/PREREGISTRATION_ton_iot.json` (written before any attacked ToN-IoT run). Calibration controls on the deduplicated data (`logs/tond_cal`): clean validation accuracy 81.1–84.9%, macro-F1 65.1–66.6%, LOSO clean FPR 0/0/0%. Trigger (`logs/select_trigger_ton.py`, rule 2; candidate set widened to non-contiguous sets after 0/96 contiguous candidates were eligible, before any attacked run): columns 7, 17, 23, 50 at 5.0, worst conditional clean rate 0%.
- Shadow (no filtering) on development seeds 43/46: the backdoor is learned (ASR 100% / 99.9%).
- Live development gate (`logs/tond_devgate`, calibration `logs/tond_cal2`): seed 43 PASSED (detection 93%, honest FPR 0%, final ASR 0.40%); **seed 46 FAILED** — attacker 2 first flagged at its 6th poisoned participation (scores 0.1, 0.8, 1.5, 2.5, 2.0, 3.1 against threshold 3.0) and poisoned retention before revocation 27.9% > 20%; final ASR nevertheless 0.06%, honest FPR 0% (honest scores ≤ 1.3 every round).
- The threshold equals the fixed alert floor (3.0, set on CICIoMT); on ToN-IoT honest scores stay ≤ 1.3, so the floor, not calibration, delays detection. On CICIoMT a floor of 2.0 raised honest FPR and cost accuracy (A21), so no single floor fits both datasets.
- Utility: final macro-F1 60.7 vs the shadow run's 66.0 on seed 46 (Ransomware recall 0% vs 90%, MITM 37.5% vs 71%) — likely the attackers' share of the rare classes (Ransomware has 208 training rows; oracle not run).
- Per the pre-registration the 5-seed confirmation does not start. No change is made pending a decision.

## A31. Data-calibrated alert floor considered and rejected (no method change)

Proposed (option B): replace the fixed alert floor 3.0 by a value derived from each dataset's clean calibration scores. Checked against the calibration files before implementing: clean (honest) score maxima are CICIoMT initial 3.21 / later 2.13 and ToN-IoT initial 2.07 / later 2.79. On ToN-IoT the clean later-phase maximum (2.79) exceeds attacker 2's undetected scores on seed 46 (2.5, 2.0), so no threshold derived from clean data can flag it earlier without also flagging clean clients at the calibrated level; on CICIoMT the same rule would lower the threshold below 3.0, which A21 showed raises honest FPR and costs accuracy. The change would not improve either dataset and was not made. bsc_7 stays frozen; the ToN-IoT gate failure (A30) stands and the ToN-IoT confirmation does not run. ToN-IoT evidence is reported at development level (two seeds), with a descriptive baseline comparison on the same seeds.

## A32. ToN-IoT descriptive comparison on development seeds 43 and 46

Runs `logs/tond_dev_baselines` (FedAvg, Krum f=2, median, trimmed mean f=2, RFA, Fed-ADH; clean and attacked), `logs/tond_dev_oracle` (evaluator-only), MediShield from `logs/tond_devgate`. Development evidence (2 seeds; not a confirmation).
- Final ASR (seed 43 / 46): FedAvg 99.99 / 99.91; median 99.95 / 99.99; trimmed mean 99.95 / 99.99; RFA 99.99 / 99.98; Fed-ADH 99.96 / 99.91; Krum 0.09 / 0.01; MediShield 0.40 / 0.06; oracle 0.29 / 0.05.
- Attacked-run macro-F1: MediShield 61.36 / 60.68 vs oracle (perfect exclusion) 61.36 / 59.14 vs FedAvg 65.98 / 66.01; Krum 57.04 / 61.44. MediShield matches perfect detection: the ~5 F1-point loss is the unavoidable cost of excluding these attackers (on seed 46 they hold the Ransomware training rows: Ransomware recall 0% for both MediShield and the oracle).
- Only Krum and MediShield remove the backdoor; Krum at a larger utility cost on seed 43 (F1 57.0, accuracy 72.8).

## A33. ToN-IoT descriptive 5-seed evaluation (declared before any run on seeds 93–97)

Because the ToN-IoT live gate failed (A30), the pre-registered confirmation cannot be claimed. A clearly labelled DESCRIPTIVE evaluation is run instead on the pre-registered seeds 93–97 with the pre-registered settings: MediShield bsc_7 clean + attacked (new `--descriptive-failed-gate` flag: filtering is authorized by the failed gate only for runs marked `COMPLETED_DESCRIPTIVE_AFTER_FAILED_GATE`, `confirmatory: false`; detector unchanged), FedAvg clean + attacked, oracle, Krum/median/trimmed mean (f = 2), RFA and Fed-ADH. `logs/evaluate_ton.py` is applied unchanged and its verdict is reported as descriptive only. Calibration refit `logs/tond_cal3` (same inputs; required by the source change; its detector core must equal `logs/tond_cal2`'s).

## A34. ToN-IoT descriptive run stopped by the absolute clean screen; rerun with the absolute floors disabled

`logs/tond_medishield` (status CLEAN_SCREEN_FAILED): seed 93's clean MediShield run flagged no client and was identical to clean FedAvg (macro-F1 68.82 both), but the runner's absolute collapse guard "every class recall ≥ 10%" failed because the IDS model itself reaches only 6.1% Backdoor recall on ToN-IoT (FedAvg also 6.1%); the runner then skipped seeds 94–97. The absolute guards (class recall ≥ 10%, macro-F1 ≥ 65) were set for CICIoMT and test the base model, not the detector. For the descriptive ToN-IoT evaluation they are disabled (`--clean-min-class-recall 0 --clean-min-macro-f1 0`); the detector-specific screen checks (honest FPR ≤ 5%, unpoisoned retention ≥ 90%) stay, and utility is judged by the pre-registered RELATIVE criteria (clean F1 vs FedAvg ≥ −1 pp, worst-class recall vs FedAvg ≥ −5 pp). Rerun into `logs/tond_medishield_b`; under the original pre-registration seed 93 counts as a clean-screen failure and that is reported.

## A35. ToN-IoT descriptive 5-seed evaluation (seeds 93–97): outcomes strong, pre-registered timing criteria not met

`logs/tond_medishield_b` (COMPLETED_DESCRIPTIVE_AFTER_FAILED_GATE), FedAvg `logs/tond_fedavg`, oracle `logs/tond_oracle`, baselines `logs/tond_baselines`; `logs/evaluate_ton.py` → `logs/ton_iot_result.json`; table `logs/comparison_ton`. Descriptive only (A30, A33, A34).
- Pre-registered criteria applied unchanged: NOT SUPPORTED (5 conclusive seeds; seed 96 passes). Failures are all about timing: an attacker first detected later than its 3rd poisoned participation (seeds 93, 94, 95, 97) and 1–3 early exposure rounds (93: round 1; 95, 97: rounds 1–3; post-exposure mean excess 2.30 pp on seed 95).
- Outcomes: final ASR 0.12 ± 0.08% vs FedAvg 98.58 ± 2.09%; detection 81.3 ± 5.1% with precision 100% (no honest update ever flagged: FPR 0% in attacked and clean runs); clean-run F1 identical to FedAvg (66.91); attacked-run F1 vs oracle −0.18 ± 0.53 pp; mean ASR over rounds 8.1% (early leak) vs FedAvg 88.3%. Defence time 1.43 s/round.
- Baselines (final ASR / clean F1): median 99.94 / 67.3; trimmed mean 99.92 / 68.9; RFA 98.91 / 65.7; Fed-ADH 98.46 / 66.7 (recall 11%, honest FPR 12.5%); Krum 20.1 ± 44.7 / 59.2 (backdoor survives on one seed; honest FPR 87.5%).
- Interpretation: on ToN-IoT the detector is slower (attackers' scores rise over the first rounds, A30) but precise, and it ends every run with the backdoor removed and utility equal to perfect detection.

## A36. One-time official CICIoMT2024 test evaluation of the frozen models

`logs/PREREGISTRATION_official_test.json` (written before any test file was read), `logs/evaluate_official_test.py`, results `logs/official_test/`. 297,371 official test rows (Benign 37,607; DDoS 154,007; DoS 98,432; Recon 3,834; MQTT 1,747; Spoofing 1,744); 154 identical to a training vector. Same preprocessing and saved training scaler; `train_defended.evaluate()` unchanged.
- **Backdoor results transfer exactly.** Main confirmation (Phase 2e seeds 71–75): triggered→Benign on the test set MediShield 0.03 ± 0.04% vs FedAvg 99.58 ± 0.57% (validation 0.29 vs 98.96). Comparison seeds: MediShield 0.01%; FedAvg, RFA, Fed-ADH 79.9% (seed 65 not learned), median 99.3%, trimmed mean 99.1%, Krum 19.8%.
- **Utility: MediShield equals FedAvg on test as on validation** (clean runs: accuracy 65.26 vs 65.32, macro-F1 60.27 vs 60.30; attacked MediShield 65.29 / 60.48).
- **Every method loses ~20 accuracy points from validation to test** (FedAvg 85.2 → 65.3). Cause: the official DoS test file (TCP_IP-DoS-ICMP) is classified as DDoS — DoS recall 0.6% on test vs 40.4% on validation for both MediShield and FedAvg — and DoS is 33% of the test rows. Other classes are close to validation (Benign 92 vs 98; DDoS 99.7; Recon 88 vs 90; MQTT 45 vs 44; Spoofing 63–66 vs 49–54). This is the base IDS's DDoS/DoS ICMP ambiguity (A20) under a capture shift, identical with and without the defence. Krum/median/trimmed mean shift the DDoS/DoS split (e.g. Krum DoS 85%, DDoS 30%), which raises their macro-F1 on test while lowering accuracy.
- Reported as is: the defence conclusions hold on the official test set; the absolute IDS accuracy on the official test set is limited by DoS-vs-DDoS generalisation in the 6-file subset.

## A37. Official-test diagnostic: DoS traffic is still detected as an attack

Read-only diagnostic after A36 (same preprocessing; models of seed 71; no reported number changed). Official DoS test rows are predicted DDoS 99.3% (FedAvg clean) / 98.9% (MediShield attacked), Benign only 0.2%. As an attack-vs-normal detector the IDS recognises 99.1% of all attack test traffic as an attack (DDoS 99.9%, DoS 99.8%, Recon 88%, MQTT 45–48%, Spoofing 60–62%) with 6.0% false alarms on normal traffic, identically with and without MediShield. The accuracy drop on the test set is a DoS-vs-DDoS labelling confusion between two attack classes, not missed attacks.

## A38. Supplementary (post-hoc) merged-Flood analysis

`logs/flood_merge_supplement.py`, results `logs/official_test_flood_supplement/`. Decided after seeing the official test results; reported as supplementary only (the pre-registered 6-class results stay primary). The same saved Phase 2e models (seeds 71–75) re-scored with DoS and DDoS merged into one Flood class (5 classes).
- Official test: accuracy 6-class 65.3% → 5-class **98.2%** for both MediShield and FedAvg (validation 85.0% → 96.9%); macro-F1 60.3 → 76.9 (validation 74.0 → 80.0); Flood recall 99.8%; attack traffic flagged as attack 99.1%; normal-traffic false alarms 7.3–7.9% (validation 1.4–2.3%).
- Remaining per-class weaknesses on test (identical with and without the defence): MQTT recall 44–45%, Spoofing 61–65%, Benign 92–93%.
- Conclusion: once the DoS/DDoS split (two ICMP-flood classes indistinguishable per flow in this subset) is removed, test accuracy matches and exceeds validation, so the 6-class test drop is that labelling ambiguity, not a failure to generalise.

## A39. Attacker-count sweep extended to 5 seeds (79–81 + 98–99)

Runs `logs/sweep5_ratio_*` (bsc_7, calibration `logs/bsc7_calR`/`bsc7_calS`, authorized by `logs/bsc7_devgate`; a first 2-attacker attempt was refused because the gate settings predate `--source-specific` — fixed by treating options absent from older gate settings as their defaults, `LIVE_GATE_LATER_DEFAULTS`; refused folder kept). Seeds 79–81 at 1 and 3 attackers are the bsc_6 runs (equivalent: no stop occurred). Table `logs/sweep5_attackers_table.csv`.
- 1 attacker: MediShield final ASR 0.04 ± 0.02% (FedAvg 79.4 ± 41.7%; seed 79 not learned); detection 96%; honest FPR 0.19%; F1 vs attacked FedAvg −0.84 ± 1.44.
- 2 attackers: 0.05 ± 0.03% (FedAvg 99.0%); detection 96.7%; honest FPR 3.33 ± 4.37% (seed 80 10%); held rounds only on seed 79; F1 −2.81 ± 2.19.
- 3 attackers: 0.17 ± 0.21% (FedAvg 99.85%); detection 92.9%; honest FPR 0%; F1 −3.07 ± 1.98 (oracle analysis A16: ≈ −2.5 is the unavoidable cost of excluding 3 of 8 clients).
- 4 of 8 attackers (50%) is outside the honest-majority assumption (also assumed by HFLMND); MediShield holds rounds and then stops (A20), which is the intended safe behaviour.

## A40. Scalability, 32 clients (alpha 0.5, 8 attackers): live gate FAILED on detection timing; backdoor removed

Runs `logs/ni_c32_*` (own clean controls). Gate (seed 46) FAILED: first detection after the 3rd poisoned participation for attackers 16 and 21 (5th) and never for attacker 26; detection 76.7%; honest FPR 0.28%; final ASR 0.09% (shadow/FedAvg-equivalent ~94%), macro-F1 72.3 (clean FedAvg on the evaluation seeds 67.5–72.3). ASR per round 93.7, 89.5, 4.8, 3.3, 0.4, then ≤ 0.2.
- Attacker scores rise over rounds 1–4 as on ToN-IoT (e.g. client 21: −0.1, 0.1, 2.1, 3.5, 5.5); honest scores reach 2.4–3.5 in rounds 1–3. Attacker 26 holds 0.3% of the data: never flagged, kept full weight, and is harmless at that size.
- Per protocol MediShield was not run on evaluation seeds 90–92 (refused). FedAvg on those seeds: attacked ASR 95.7–95.9%, clean F1 67.5–72.3.

## A41. 32 clients: descriptive evaluation (seeds 90–92)

`logs/ni_c32_medishield` (COMPLETED_DESCRIPTIVE_AFTER_FAILED_GATE; first attempt refused, kept as `ni_c32_medishield_refused`), FedAvg `logs/ni_c32_fedavg`.
- Final ASR 0.01 / 0.07 / 0.03% vs FedAvg 95.85 / 95.83 / 95.67%; mean ASR over rounds 7.3 / 12.7 / 7.3% (early leak before the slower detections, as in A40).
- Detection 46.7 / 60.8 / 53.3% (the smallest attackers are flagged late or never); honest FPR ≤ 0.28% in attacked and clean runs; no held rounds or stops.
- Clean-run F1 equal to FedAvg on seeds 90–91 (72.31 / 71.59 vs 72.31 / 71.61; seed 92 71.68 vs 67.49, DDoS/DoS swap). Attacked-run F1 vs attacked FedAvg −4.4 / −0.9 / −4.3 pp (8 of 32 clients excluded; oracle not run).
- Server time per round (laptop wall clock): probe measurement 0.94 s (8 clients), 1.22 s (16), 9.17 s (32); decision 0.05 / 0.08 / 0.82 s; local training 9.2 / 5.9 / 24.7 s. The 32-client run was slower than linear for training too, so machine load contributes; the measurement cost is linear in the number of clients by construction (one probe pass per submitted model).
