# Issue tracker: Local Markdown

Issues and specs for this repo are markdown files in `.scratch/`.

## Conventions

- Each feature has one directory: `.scratch/<feature-slug>/`.
- The spec is `.scratch/<feature-slug>/spec.md`.
- Each implementation ticket is one file: `.scratch/<feature-slug>/issues/<NN>-<slug>.md`. The numbers start at `01`. Never put all the tickets in one combined file.
- A `Status:` line near the top of each issue file records the triage state. `triage-labels.md` gives the role strings.
- Comments and conversation history go at the bottom of the file, under a `## Comments` heading.

## When a skill says "publish to the issue tracker"

Create a new file under `.scratch/<feature-slug>/`. Create the directory if it does not exist.

## When a skill says "fetch the relevant ticket"

Read the file at the given path. The user usually gives the path or the issue number.

## A change that also needs burn-mamba

This tracker holds only the work of this repo. `../burn-mamba` and `../burn-stack` stay read-only from a session of this repo, because they have other users.

- **The burn-mamba part**: a ticket in the burn-mamba tracker (`../burn-mamba/docs/agents/issue-tracker.md`). A session of this repo can write that ticket: the access allowlist of `CLAUDE.md` permits writes to the `.scratch/` of burn-mamba and burn-stack. A burn-mamba session does the change. A change to `../burn-stack` follows the same steps, with the burn-stack tracker.
- **Blocking**: the `Blocked by:` line of a ticket here gives the path of the burn-mamba ticket.
- **Review**: `/code-review` here reads only the diff of this repo.

### The burn-mamba pin

`Cargo.toml` pins `burn-mamba` to a GitHub rev, and takes `burn` from crates.io. The `burn` line must give the same `burn` as the active `burn` line of `../burn-mamba`. That is the same crates.io version, or the same git rev when `../burn-mamba` uses a burn git rev. Else cargo resolves two `burn` crates, and no types unify. While the burn-mamba part is in progress, use one of these:

- **No active burn-mamba session or parallel agent** (`ListAgents` shows them): enable the commented `path` line of `burn-mamba` in `Cargo.toml`. The build then uses the working tree of `../burn-mamba`.
- **An active burn-mamba session or parallel agent**: it can change that working tree. Wait for its push, then move the pin to the pushed commit.

Each commit of this repo builds against the pin. Before you commit in this repo:

1. Set the `burn-mamba` rev to the pushed burn-mamba commit.
2. Set the `burn` line to the active `burn` line of `../burn-mamba` at that commit.
3. Comment out the `path` lines, and enable the `burn-mamba` git line.
4. Run `cargo check`, which updates `Cargo.lock`.

## Wayfinding operations

`/wayfinder` uses these. The **map** is one file, with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`). A `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when each file in that line is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for the files that are open, unblocked and unclaimed. The lowest number wins.
- **Claim**: set `Status: claimed` and save the file before any other work.
- **Resolve**: write the answer under an `## Answer` heading and set `Status: resolved`. Then add a context pointer (gist + link) to the Decisions-so-far of `map.md`.
