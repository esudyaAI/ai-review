"""CLI entrypoint for the reviewer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

from reviewer.adjudication import adjudicate
from reviewer.config import load_config
from reviewer.deterministic import run_deterministic
from reviewer.diff import DiffTooLarge, check_limits
from reviewer.findings import keep_blocking
from reviewer.gate import decide
from reviewer.gitlab_client import GitLabClient
from reviewer.llm.loop import ReviewIncomplete, ReviewLoop
from reviewer.llm.openai_provider import OpenAIProvider
from reviewer.prompt_builder import build_system_prompt, build_user_prompt
from reviewer.tools import RepoTools

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_TECHNICAL = 2
EXIT_TOO_LARGE = 3


def review(config, repo: Path, client, make_reviewer, adjudicator, mr_iid: int) -> int:
    try:
        mr = client.fetch_diff(mr_iid)
        check_limits(mr, config.limits)
    except DiffTooLarge as exc:
        print(f"AI review: {exc}", file=sys.stderr)
        return EXIT_TOO_LARGE
    except Exception as exc:  # noqa: BLE001 - any failure here is technical, not a verdict
        print(f"AI review: техническая ошибка при получении диффа: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL

    try:
        claude_md = (repo / "CLAUDE.md").read_text(encoding="utf-8") if (repo / "CLAUDE.md").exists() else ""
        findings = list(run_deterministic(repo, config.linter, mr.changed_files))
        # High-risk paths in the diff pull the whole review to the stronger model.
        reviewer = make_reviewer(config.model_for(mr.changed_files))
        findings.extend(reviewer.run(build_system_prompt(claude_md), build_user_prompt(mr)))
    except ReviewIncomplete as exc:
        print(f"AI review: ревью не завершено ({exc}), нужен реран", file=sys.stderr)
        return EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        print(f"AI review: техническая ошибка ревью: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL

    blocking = keep_blocking(findings)

    try:
        discussions = client.list_discussions(mr_iid)
        decision = decide(blocking, discussions, mr.changed_files)

        for finding in decision.to_create:
            client.create_discussion(mr_iid, finding, mr)
        for discussion_id in decision.to_resolve:
            client.resolve(mr_iid, discussion_id, "Замечание исправлено, тред закрыт автоматически.")

        accepted: set[str] = set()
        by_id = {d.id: d for d in discussions}
        for discussion_id in decision.to_adjudicate:
            verdict = adjudicator(by_id[discussion_id])
            if verdict.accepted:
                client.resolve(mr_iid, discussion_id, f"Аргумент автора принят: {verdict.reason}")
                accepted.add(discussion_id)
            else:
                client.reply(mr_iid, discussion_id, f"Замечание остаётся: {verdict.reason}")
    except Exception as exc:  # noqa: BLE001 - a mid-loop GitLab failure is technical, not a verdict
        print(f"AI review: техническая ошибка при работе с GitLab: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL

    blocking_left = decision.blocking_count - len(accepted)
    print(f"AI review: новых замечаний {len(decision.to_create)}, "
          f"закрыто {len(decision.to_resolve) + len(accepted)}, "
          f"осталось открытых {blocking_left}")
    return EXIT_FINDINGS if blocking_left > 0 else EXIT_CLEAN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reviewer")
    parser.add_argument("command", choices=["review"])
    parser.add_argument("--mr", type=int, required=True)
    args = parser.parse_args(argv)

    repo = Path(os.environ.get("CI_PROJECT_DIR", "."))
    config = load_config(repo / ".ai-review.yml")
    api_key = os.environ["OPENAI_API_KEY"]

    def make_provider(model: str) -> OpenAIProvider:
        return OpenAIProvider(api_key=api_key, base_url=config.base_url, model=model)

    client = GitLabClient(
        api_url=os.environ["CI_API_V4_URL"],
        token=os.environ["AI_REVIEW_GITLAB_TOKEN"],
        project_id=os.environ["CI_PROJECT_ID"],
        client=httpx.Client(timeout=60.0),
    )
    tools = RepoTools(root=repo)
    return review(
        config,
        repo,
        client,
        lambda model: ReviewLoop(make_provider(model), tools),
        lambda discussion: adjudicate(make_provider(config.model), discussion),
        args.mr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
