Generating the docs
----------

This repository uses MkDocs with the Material theme. The MkDocs configuration is
located at `docs/mkdocs.yaml` and the documentation sources are under
`docs/source/`.

Build locally from the repository root:

    mkdocs build --config-file docs/mkdocs.yaml

Serve locally (auto-rebuild on changes):

    mkdocs serve --config-file docs/mkdocs.yaml
