# Fed-MediShield

Code for the paper **"Fed-MediShield: Behaviour-Probing Defence Against Backdoor Attacks in Federated
Intrusion Detection for the Internet of Medical Things"** (Akshay K, IIT Patna).

Fed-MediShield is a server-side defence for federated intrusion detection systems (IDS). It inspects what
each client update *does* to the protected decision (attack traffic classified as benign) rather than how
the update looks. Every round, the server:

1. probes each submitted model on a small trusted reference sample (588 feature probes, two folds);
2. scores how broadly the update pushes attack traffic of many classes towards *benign*, relative to the
   client's peers and to its own known-clean profile, so that honest clients with unusual (non-IID) data
   are rarely flagged;
3. turns the score into bounded aggregation weights (0 ≤ a_i ≤ p_i), with norm clipping, revocation of
   the undetected first poisoned contribution, and a hold rule when too many clients are flagged.

The server never sees attacker identities, the trigger, poison masks or test data.

## Main results

| Setting | FedAvg (no defence) | Fed-MediShield |
|---|---|---|
| CICIoMT2024, three confirmation batches (15 fresh seeds, 12 conclusive): final attack success rate | 99.09% | **0.17%** |
| CICIoMT2024, official test split: attack success rate | 99.58% | **0.03%** |
| ToN-IoT, 5 seeds (descriptive): attack success rate | 98.58% | **0.12%** |
| Clean-run macro-F1 minus FedAvg (15 seeds) | – | −0.55 points (95% CI [−1.47, 0.38]) |

Two of the three confirmation batches met all criteria fixed in advance; the batch on seeds 82–86 did not.
Among FedAvg and eight robust baselines (Krum, Multi-Krum, median, trimmed mean, RFA, Fed-ADH, FLTrust,
FLAME) it was the only method that removed the backdoor while preserving utility.

**Where it fails.** An adaptive attacker who knows the probe design, camouflages probe-like inputs and places
the trigger outside the probe bank evades the defence completely, and a larger secret probe bank did not
reliably prevent this. Detection is also slower on ToN-IoT and with 32 clients, the method fails under
extreme heterogeneity (Dirichlet α = 0.1), and source-specific backdoors partly evade it.

## Repository layout

| Path | Contents |
|---|---|
| `src/medishield_candidate.py` | **The method** (version `medishield_bsc_7`): calibration, gates, live defence, oracle reference, self-checks |
| `src/train_defended.py` | Federated training runner: data loading, partition, MLP, local training, poisoning, metrics; also the baselines |
| `src/defenses.py` | Baseline aggregators (Krum, Multi-Krum, median, trimmed mean, RFA, Fed-ADH, FLTrust, FLAME) |
| `src/attack.py` | Trigger and poisoning helpers |
| `src/prepare_federated.py` | Builds the CICIoMT2024 subset used in the paper |
| `data/prepare_ton_iot.py` | Builds the ToN-IoT network data used in the paper |
| `data/processed_*/` | Feature lists and partition manifests of the processed data (the data itself is not included) |
| `logs/PREREGISTRATION_*.json`, `PLAN_*`, `AMENDMENT_*` | Criteria fixed **before** each evaluation run |
| `logs/*_result.json`, `logs/*/summary.csv` | Recorded verdicts and summary tables behind the paper |
| `logs/evaluate_*.py`, `compare_methods.py`, `make_figure*.py` | Evaluation, comparison-table and figure scripts |
| `figures/` | All figures (paper Figs. 1–4 and supplementary ones) |
| `paper/` | LaTeX source of the paper |
| `project_info/` | Full development record, including failed attempts (`operations.md`, `operations_addendum_2026-10.md`) |
| `archive/early_pipeline/` | Early-stage scripts kept for the record; not used by the final pipeline |

**About the file name:** the method file keeps its development name `medishield_candidate.py` because the
pre-registration files and every run record refer to it by that name.

## Installation

Tested with Python 3.14.7 on Windows 11, CPU only.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Quick check (about 15 seconds, synthetic data only, no dataset needed):

```bash
python src/medishield_candidate.py --self-check
```

It should end with `ALL SELF-CHECKS PASSED`.

## Data

The datasets are not redistributed here; download them from their owners.

**CICIoMT2024** (Canadian Institute for Cybersecurity, University of New Brunswick). Put these six training
files in `data/raw/`:

```
Benign_train.pcap.csv               TCP_IP-DDoS-ICMP1_train.pcap.csv   TCP_IP-DoS-ICMP1_train.pcap.csv
Recon-OS_Scan_train.pcap.csv        MQTT-Malformed_Data_train.pcap.csv ARP_Spoofing_train.pcap.csv
```

For the official-test evaluation, also put the matching `*_test.pcap.csv` files in `data/raw/`
(they are never read during training or calibration). Then:

```bash
python src/prepare_federated.py --raw-dir data/raw --out data/processed_grouped
```

**ToN-IoT** (UNSW Canberra). Put `train_test_network.csv` (the network train/test file) in `data/raw_ton/`, then:

```bash
python data/prepare_ton_iot.py --raw data/raw_ton/train_test_network.csv --out data/processed_ton_dedup --dedup
```

Both scripts use seed 42, 8 clients and Dirichlet α = 0.5 by default, as in the paper. Compare the
`partition_manifest.json` they write with the one in this repository to confirm you built the same data.

## Reproducing the experiments

All commands run from the repository root. Each step writes a folder under `logs/`, and an interrupted
step continues if you repeat the identical command with `--resume`.

> **Important:** the code's built-in defaults are not the paper's settings. Always pass the six flags below
> (shown as `$PAPER`). On Windows PowerShell, replace the `\` at line ends with a backtick, or write each
> command on one line.

```bash
PAPER="--loss-adjustment logit --logit-tau 0.5 --history-channel off --trigger-features 5 6 7 8 --poison-rate 0.5 --malicious-ratio 0.25"
```

The protocol separates **development seeds** (43, 46; used to check the detector) from **fresh evaluation
seeds** (71–75), and calibration seeds (42, 44, 45) from both. The code refuses to filter with a detector
that has not passed a gate.

**1. Calibrate on clean training runs.** Trains clean FedAvg runs on the calibration seeds and fits the
thresholds. This first calibration reports a failed gate because no attacked development run exists yet;
that is expected.

```bash
python src/medishield_candidate.py --stage calibrate --train-missing-controls $PAPER --out logs/cal_initial
```

**2. Record an attacked development run in shadow mode.** Shadow mode trains with plain FedAvg; the
detector only observes.

```bash
python src/medishield_candidate.py --stage evaluate --calibration logs/cal_initial/calibration/calibration.json \
    --shield-mode shadow --condition attacked --seeds 43 46 $PAPER --out logs/shadow_dev
```

**3. Recalibrate with the recorded runs.**

```bash
python src/medishield_candidate.py --stage calibrate --capture-source logs/cal_initial/calibration_controls logs/shadow_dev \
    --reuse-measurements logs/cal_initial $PAPER --out logs/cal_final
```

**4. Live development gate** (the defence is switched on for the development seeds only):

```bash
python src/medishield_candidate.py --stage evaluate --calibration logs/cal_final/calibration/calibration.json \
    --development-gate-run --condition attacked --seeds 43 46 $PAPER --out logs/devgate
```

Continue only if it reports `DEVELOPMENT LIVE GATE: PASSED`.

**5. Evaluate on fresh seeds** (clean and attacked runs):

```bash
python src/medishield_candidate.py --stage evaluate --calibration logs/cal_final/calibration/calibration.json \
    --live-gate-dir logs/devgate --condition both --seeds 71 72 73 74 75 $PAPER --out logs/phase2e_medishield
```

**6. References on the same seeds:** undefended FedAvg, the baselines, and the oracle (an evaluator-only
reference that removes the true attackers, i.e. perfect detection; not a deployable defence).

```bash
python src/train_defended.py --defense fedavg --condition both --rounds 15 --seeds 71 72 73 74 75 \
    --loss-adjustment logit --logit-tau 0.5 --trigger-features 5 6 7 8 --poison-rate 0.5 --malicious-ratio 0.25 \
    --out logs/phase2e_fedavg
python src/train_defended.py --defenses krum median trimmed_mean rfa fed_adh --f 2 --condition both --rounds 15 \
    --seeds 71 72 73 74 75 --loss-adjustment logit --logit-tau 0.5 --trigger-features 5 6 7 8 --poison-rate 0.5 \
    --malicious-ratio 0.25 --out logs/phase2e_baselines
python src/medishield_candidate.py --stage oracle --condition attacked --oracle-aggregation residual \
    --seeds 71 72 73 74 75 $PAPER --out logs/phase2e_oracle
```

**7. Comparison table** (attack success rate, macro-F1, detection recall and false-alarm rate per method):

```bash
python logs/compare_methods.py --medishield logs/phase2e_medishield --seeds 71 72 73 74 75 \
    --baselines logs/phase2e_fedavg logs/phase2e_baselines --out logs/my_comparison
```

**ToN-IoT.** Repeat steps 1–7 with `--data data/processed_ton_dedup/federated_data.pt`, trigger features
`7 17 23 50` instead of `5 6 7 8`, different output folder names, and evaluation seeds 93–97. In our
experiments the ToN-IoT live gate failed on detection timing, so the paper reports these runs as
**descriptive**: they were run with `--descriptive-failed-gate --clean-min-class-recall 0 --clean-min-macro-f1 0`
added to step 5 (the base IDS itself has under 10% recall on one ToN-IoT class).

**Further baselines.** Multi-Krum, FLTrust and FLAME are run like step 6, e.g.
`--defenses multi_krum fltrust flame --f 2` on seeds 63–67. `--fltrust-loss logit` gives FLTrust's server
the clients' logit-adjusted loss; `--flame-lambda` sets FLAME's noise level (default 0.001).

**Adaptive attacks and the probe-bank countermeasure.** These options exist in both runners:

| Option | Meaning |
|---|---|
| `--probe-camouflage 1.0` | every unpoisoned attack row of an attacker gets one random probe-style perturbation and keeps its true label |
| `--trigger-features 2 8 15 36 --trigger-value 4.5` | the out-of-bank trigger (columns no contiguous probe group covers) |
| `--camouflage-random-groups 1000` | the attacker mirrors the larger bank design in its camouflage |
| `--random-groups 1000 --probe-seed 50821` | the defender's secret large bank (4508 probes); needs a new calibration and development gate |

The exact commands, seeds and decision rules for these runs are in `logs/PLAN_extra_baselines_adaptive.json`,
`logs/AMENDMENT_adaptive_extension.json` and `logs/PLAN_review2_experiments.json`, each committed before
its runs.

**Run time.** On our laptop CPU one round of a Fed-MediShield run took about 7–28 s (8 clients trained one
after another, plus about 0.7 s of server-side defence), so one 15-round run takes a few minutes. The
calibration steps take longer because every round of every calibration run is probed.

**Determinism.** Seeds are fixed and PyTorch deterministic mode is on, so results should repeat closely.
Small floating-point differences across machines or library versions can still change individual
decisions near the threshold.

## Recorded results behind the paper

| Paper item | Recorded file |
|---|---|
| Confirmation, seeds 71–75 (supported) | `logs/phase2e_result.json` (criteria: `logs/PREREGISTRATION_bsc6.json`) |
| Confirmation, seeds 82–86 (not supported) | `logs/phase2f_result.json` (criteria: `logs/PREREGISTRATION_bsc7.json`) |
| Confirmation, seeds 107–111 (supported) | `logs/main_extension_result.json` (criteria: `logs/PLAN_review2_experiments.json`) |
| Table I (method comparison) | `logs/comparison_bsc6_phase2d_seeds/`, `logs/comparison_ton/`, `logs/comparison_extra*_cic/`, `logs/comparison_extra*_ton/` |
| Adaptive attacks | `logs/adaptive_result.json` |
| Secret large probe bank | `logs/bigbank_result.json` |
| ToN-IoT | `logs/ton_iot_result.json` |
| Official test split | `logs/official_test/summary.csv` |
| DoS + DDoS merged (post-hoc supplement) | `logs/official_test_flood_supplement/summary.csv` |
| Robustness variants | `logs/stage3_bsc6_result.json`, `logs/stage3_variants_result.json`, `logs/stage3b_result.json` |
| Sensitivity sweep | `logs/hparams_table/hparams_per_seed.csv` |
| Heterogeneity and scale | `figures/table_noniid_scalability.csv` |
| Server run time | `logs/runtime_table/runtime_summary.csv` |
| Trigger selection (before any attacked run) | `logs/trigger_selection_*/selection.json` |

The `logs/evaluate_*.py` scripts are the pre-registered verdict scripts. They check our recorded runs
against the code and calibration fingerprints fixed in the pre-registration files, so they will not accept
newly produced runs; for your own runs use `compare_methods.py` (step 7) or `--stage report`.

The figure and evaluation scripts read the full per-run folders, which are too large for this repository.
The per-run result bundles (about 140 MB) will be published separately; the link will be added here.

**Versions.** The method file is `medishield_bsc_7`. The main CICIoMT2024 runs, Table I and the official
test were run with `medishield_bsc_6`, which differs only in stopping a run at the first invalid round
instead of holding it. None of those runs had an invalid round, so both versions give identical results
there (see the Protocol paragraph of the paper). Seeds 82–86, 107–111 and all later runs used
`medishield_bsc_7`.

**Release.** The paper cites release `v1.0` (commit `9728b17`), which holds the code, protocol files and
recorded results used for the paper. Later commits only update the paper source and this README.

## Scope and limitations

This is research code. It is not robust by guarantee: thresholds are empirical quantiles, the probe bank
overlaps the synthetic trigger family used in the experiments, an attacker who knows the probe design can
evade it (see "Where it fails" above), and the breadth statistic assumes all-to-one backdoors. The experiments use a six-file subset
of CICIoMT2024, an MLP, feature-space triggers, 15 rounds and up to 32 clients. The server needs plaintext
updates and a trusted reference sample, which conflicts with secure aggregation.

## Citation

See `CITATION.cff`. Publication details will be added once the paper is published.

## License

MIT; see `LICENSE`.
