# Documentation deployment

Build and preview the existing MkDocs site from the repository root:

```bash
uv sync --locked --extra docs --python 3.11
uv run --extra docs mkdocs build --strict
uv run --extra docs mkdocs serve
```

Open `http://127.0.0.1:8000`. Documentation generation does not require RatesMC,
TALYS, or the HFB dataset.

## GitHub Pages setup

When the repository is ready for publication:

1. In GitHub, open **Settings → Pages** and select **GitHub Actions** as the source.
2. Push the documentation workflow and configuration to `main`.
3. Confirm that **Actions → Documentation** completes successfully.
4. Open the published URL shown in the `github-pages` deployment environment.

The workflow checks pull requests and deploys successful builds from `main`.
It can also be started with **Actions → Documentation → Run workflow**.
Site and repository URLs use the current repository name automatically. For a
custom domain, update `DOCS_SITE_URL` in the workflow.
