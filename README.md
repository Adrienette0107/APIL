# APIL

APIL is a deployable FastAPI middleware layer. It extracts Prompt DNA and
optimizes requests before routing them to a configured AI provider. The APIL
container does not bundle a GenAI model; providers and model runtimes remain
external dependencies.

## Deploy with Docker Compose

1. Put production secrets in `.env` or your deployment secret store.
2. Set `DATABASE_URL` to a reachable PostgreSQL database.
3. Set `OLLAMA_CONTAINER_URL` only when using Ollama. For host Ollama on Docker
    Desktop, `host.docker.internal` is the default container-facing address.
4. Start APIL:

```powershell
docker compose up --build -d
```

The service listens on `http://localhost:8000` and exposes Swagger at
`/docs`. Liveness is `/health`; readiness, including the database check, is
`/ready`.

Provider selection is controlled by the existing `model` request field and
provider configuration. The container does not require a paid API provider or
an Ollama container to start.
