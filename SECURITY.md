# Security and private configuration

Never commit runtime `.env` files, API keys, passwords, SSH private keys,
Hugging Face tokens, model weights, or machine-specific logs.

Only the `*.env.example` templates belong in source control. The install
scripts create ignored `config/head.env` and `config/worker.env` files for real
settings.

Before publishing a release, run:

```bash
./scripts/check-private-data.sh
```

The published container does not include model weights. Model and cache paths
are bind-mounted at runtime.
