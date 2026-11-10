# Code review

The additions of this repo to `/code-review`.

## What each agent receives

- **Standards agent:** the **Constraints** and **Names** sections of the ticket, when the ticket has them. A constraint of the ticket wins over a general rule of `CLAUDE.md`. For example, a ticket that permits a new dependency stops the "reach for a Burn crate first" rule for that dependency.
- **Both agents:** the results of the checks that already ran: the `cargo check` of each feature set that the change reaches (the bacon jobs), and `cargo test`. Each agent uses these results as they are, and it reports what the checks cannot see.

## Review before the last commit

A ticket with a fixed list of commits holds the fixes of the review in its last commit. The diff of `/code-review` ends at `HEAD`, so it does not show uncommitted work. Before the review:

1. Run `git add -N <path>` for each new file. This records only the path, not the content.
2. Give each agent the command `git diff <base>`, in place of `git diff <base>...HEAD`.

The review can start when `git diff <base> --stat` lists each changed file.
