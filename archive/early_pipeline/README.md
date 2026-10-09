# Early pipeline (archived)

These scripts come from the first stage of the project: a centralised baseline, plain federated training,
a first backdoor experiment and a diagnostic audit of an earlier detector version. The final pipeline in
`src/` does not import or call any of them, and no result in the paper depends on them.

They are kept unchanged for the record. They were written to run from `src/`, so their imports
(for example `from train_centralized import ...`) only work if a file is copied back there.

| File | What it was |
|---|---|
| `data.py` | early data loader for the centralised baseline |
| `logger.py` | plotting/logging helper of the early scripts |
| `train_centralized.py` | centralised MLP baseline |
| `train_federated.py` | plain FedAvg training |
| `train_attacked.py` | first backdoor (trigger) experiment |
| `medishield_audit.py` | read-only diagnostic of saved measurements from an earlier detector version |
