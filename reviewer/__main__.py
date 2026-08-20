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
from reviewer.llm.usage import (
    ADJUDICATION_PHASE,
    REVIEW_PHASE,
    UsageTracker,
    merge_loop_stats,
    report_lines,
)
from reviewer.prompt_builder import build_system_prompt, build_user_prompt
from reviewer.testlayout import describe_test_layout, detect_test_globs
from reviewer.tools import RepoTools

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_TECHNICAL = 2
EXIT_TOO_LARGE = 3


def review(config, repo: Path, client, make_reviewer, adjudicator, mr_iid: int, on_loop=None) -> int:
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
        findings = list(run_deterministic(repo, mr.changed_files))
        # Each half of the diff is reviewed by the model its own risk deserves.
        # Routing per merge request instead of per file meant one auth file made
        # the strong model read all nine files of jobbytime-go !29 at $30/M output.
        high_risk_files, other_files = config.split_by_risk(mr.changed_files)
        passes = [
            (config.model_high_risk, high_risk_files),
            (config.model, other_files),
        ]
        system_prompt = build_system_prompt(claude_md, describe_test_layout(repo))
        for model, files in passes:
            if not files:
                continue
            reviewer = make_reviewer(model)
            # The caller wants the loop's counters even if the review below raises:
            # a run that died after twenty tool calls is exactly the case worth seeing.
            if on_loop is not None:
                on_loop(reviewer)
            findings.extend(reviewer.run(system_prompt, build_user_prompt(mr.subset(files))))
    except ReviewIncomplete as exc:
        print(f"AI review: ревью не завершено ({exc}), нужен реран", file=sys.stderr)
        return EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        print(f"AI review: техническая ошибка ревью: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL

    blocking = keep_blocking(findings)

    try:
        discussions = client.list_discussions(mr_iid)
        decision = decide(blocking, discussions, mr.changed_files, detect_test_globs(repo),
                          mr.added_lines_by_file)
    except Exception as exc:  # noqa: BLE001 - a GitLab failure here is technical, not a verdict
        print(f"AI review: техническая ошибка при работе с GitLab: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL

    # Each mutating call is isolated: a GitLab failure on one finding (network blip, 500,
    # etc.) must not abort the whole run and hide every other finding. The exit code still
    # comes from decision.blocking_count/accepted below, so a real blocking finding still
    # blocks the merge even if we failed to post/resolve its GitLab thread.
    created_count = 0
    for finding in decision.to_create:
        try:
            client.create_discussion(mr_iid, finding, mr)
            created_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"AI review: не удалось опубликовать находку {finding.fingerprint}: {exc}", file=sys.stderr)

    resolved_count = 0
    for discussion_id in decision.to_resolve:
        try:
            client.resolve(mr_iid, discussion_id, "Замечание исправлено, тред закрыт автоматически.")
            resolved_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"AI review: не удалось закрыть тред {discussion_id}: {exc}", file=sys.stderr)

    accepted: set[str] = set()
    by_id = {d.id: d for d in discussions}
    for discussion_id in decision.to_adjudicate:
        try:
            verdict = adjudicator(by_id[discussion_id])
            if verdict.accepted:
                client.resolve(mr_iid, discussion_id, f"Аргумент автора принят: {verdict.reason}")
                accepted.add(discussion_id)
                resolved_count += 1
            else:
                client.reply(mr_iid, discussion_id, f"Замечание остаётся: {verdict.reason}")
        except Exception as exc:  # noqa: BLE001
            print(f"AI review: не удалось обработать адъюдикацию треда {discussion_id}: {exc}", file=sys.stderr)

    # blocking_count/exit code reflect the real gate decision regardless of whether
    # publishing to GitLab succeeded (see the isolated try/except above) - a genuine
    # blocking finding still blocks the merge even if we failed to post its thread.
    # created_count/resolved_count in the summary line below are the actual counts of
    # successful GitLab mutations, not the planned ones, so the printed summary matches
    # what a reviewer will actually see in the merge request.
    # A suppressed finding must never be silent: the whole point of the scope rule
    # is that it can hide a real, different problem in the same file.
    for category, path, line in decision.outside_diff:
        print(f"AI review: находка [{category}] в {path}:{line} отброшена — "
              f"строка вне изменённого диффа этого merge request'а")

    for category, path in decision.covered_by_new_tests:
        print(f"AI review: находка [{category}] в {path} отброшена — этот merge "
              f"request добавляет тесты в том же пакете, значит покрытие он не ухудшал")

    for category, path in decision.not_applicable:
        print(f"AI review: находка [{category}] в {path} отброшена — "
              f"в этом файле нечего вызывать, категория неприменима")

    for category, path in decision.folded:
        print(f"AI review: находка [{category}] в {path} отнесена к уже открытому "
              f"треду той же категории в этом файле, отдельный тред не заводится")

    for category, path in decision.suppressed:
        print(f"AI review: находка [{category}] в {path} подавлена — "
              f"по этой паре (категория, файл) аргумент автора уже принят")

    blocking_left = decision.blocking_count - len(accepted)
    print(f"AI review: новых замечаний {created_count}, "
          f"закрыто {resolved_count}, "
          f"осталось открытых {blocking_left}")
    return EXIT_FINDINGS if blocking_left > 0 else EXIT_CLEAN


def _adjudicator_for(config, make_provider):
    """Build the adjudication callback, routing to config.model_for(discussion.file) so a
    dispute over a high-risk finding (e.g. billing/migrations) is resolved by the same
    stronger model that would have produced it, not always the base model."""

    def adjudicate_discussion(discussion):
        model = config.model_for([discussion.file] if discussion.file else [])
        return adjudicate(make_provider(model, ADJUDICATION_PHASE), discussion)

    return adjudicate_discussion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reviewer")
    parser.add_argument("command", choices=["review"])
    parser.add_argument("--mr", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        repo = Path(os.environ.get("CI_PROJECT_DIR", "."))
        config = load_config(repo / ".ai-review.yml")
        api_key = os.environ["OPENAI_API_KEY"]
        usage_tracker = UsageTracker()

        def make_provider(model: str, phase: str = REVIEW_PHASE) -> OpenAIProvider:
            return OpenAIProvider(
                api_key=api_key,
                base_url=config.base_url,
                model=model,
                usage_tracker=usage_tracker,
                phase=phase,
            )

        client = GitLabClient(
            api_url=os.environ["CI_API_V4_URL"],
            token=os.environ["AI_REVIEW_GITLAB_TOKEN"],
            project_id=os.environ["CI_PROJECT_ID"],
            client=httpx.Client(timeout=60.0),
        )
        tools = RepoTools(root=repo)
        loops: list[ReviewLoop] = []
        code = review(
            config,
            repo,
            client,
            lambda model: ReviewLoop(make_provider(model), tools),
            _adjudicator_for(config, make_provider),
            args.mr,
            on_loop=loops.append,
        )
        stdout_lines, stderr_warnings = report_lines(
            usage_tracker.by_model(),
            config.pricing,
            by_phase=usage_tracker.by_phase(),
            loop_stats=merge_loop_stats(loop.stats for loop in loops),
        )
        for line in stdout_lines:
            print(line)
        for warning in stderr_warnings:
            print(warning, file=sys.stderr)
        return code
    except Exception as exc:  # noqa: BLE001 - startup/config/env failures are technical, not a verdict
        print(f"AI review: техническая ошибка при старте: {exc}", file=sys.stderr)
        return EXIT_TECHNICAL


if __name__ == "__main__":
    raise SystemExit(main())
