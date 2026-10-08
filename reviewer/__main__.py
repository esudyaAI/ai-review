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
from reviewer.reads import encode_marker
from reviewer.testlayout import describe_test_layout, detect_test_globs
from reviewer.tools import RepoTools

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_TECHNICAL = 2
EXIT_TOO_LARGE = 3

# What the log says about each exit code. The codes already told the two failures
# apart - 1 is "there are blocking findings", 2 is "the review did not happen" -
# but a person opening a red job saw only the exception text, and in the list of
# jobs both look identically failed. So every run ends with one line that names
# the outcome in words next to its code.
#
# 2 and 3 are both "the review did not happen" and are fixed differently: 2 by a
# retry or by fixing the infrastructure, 3 by splitting the merge request. A single
# shared wording would hide that.
_OUTCOME_BY_CODE = {
    EXIT_CLEAN: "замечаний нет",
    EXIT_FINDINGS: "есть блокирующие замечания",
    EXIT_TECHNICAL: "ревью не состоялось из-за технической ошибки, помогает повтор",
    EXIT_TOO_LARGE: "ревью не состоялось: дифф больше лимита, merge request надо разбить",
}


def outcome_line(code: int) -> str:
    return f"AI review: итог — {_OUTCOME_BY_CODE.get(code, 'неизвестный исход')}, код выхода {code}"


def _one_line(text: str, limit: int = 600) -> str:
    """A finding's text as one log line.

    The job log is grepped by the "AI review:" prefix, so a multi-line problem
    statement must not spill onto lines that lack it. Cut rather than dropped
    when long: the start of a finding is what identifies it.
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _describe(exc: BaseException) -> str:
    """Name the exception class, not just its text.

    Measured on httpx 0.28: str() of ReadTimeout, ConnectTimeout, ConnectError and
    RemoteProtocolError all return the empty string. Every network failure therefore
    printed "техническая ошибка при получении диффа: " and stopped at the colon,
    which says nothing about what to fix. The class name is the only thing that
    survives such an exception, so it goes first.
    """
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else f"{type(exc).__name__} (без текста)"


def _drop_generated(mr, config):
    """The merge request without its generated files, or None if that is all it was.

    Generated files leave before `check_limits`, not after it: a merge request
    of a builder and the file that builder produces used to be refused whole
    for being too large, so the builder -- the part with decisions in it --
    went unreviewed. They leave before the prompt too, because there is nothing
    in them to judge and the tokens are paid for.

    Nothing is dropped quietly: the files are named in the log either way. An
    empty diff is left alone -- it is a strange state, not a merge request of
    generated files, and exiting zero on it would hide that.
    """
    if not mr.files:
        return mr
    reviewable, generated = config.partition_generated(mr.changed_files)
    if not generated:
        return mr
    print(
        f"AI review: не ревьюируются, собираются из репозитория: "
        f"{', '.join(generated)} (generated_paths в .ai-review.yml)",
        file=sys.stderr,
    )
    if not reviewable:
        print(
            "AI review: пропущено, в merge request нет ничего, кроме собранных файлов",
            file=sys.stderr,
        )
        return None
    return mr.subset(reviewable)


def review(config, repo: Path, client, make_reviewer, adjudicator, mr_iid: int, on_loop=None) -> int:
    try:
        mr = client.fetch_diff(mr_iid)
        mr = _drop_generated(mr, config)
        if mr is None:
            return EXIT_CLEAN
        check_limits(mr, config.limits)
    except DiffTooLarge as exc:
        print(f"AI review: {exc}", file=sys.stderr)
        return EXIT_TOO_LARGE
    except Exception as exc:  # noqa: BLE001 - any failure here is technical, not a verdict
        print(f"AI review: техническая ошибка при получении диффа: {_describe(exc)}", file=sys.stderr)
        return EXIT_TECHNICAL

    # Отмена до первого обращения к модели. Здесь, а не в rules: CI-шаблона:
    # changes: в GitLab умеет только «запустить, если задет один из этих путей»,
    # то есть потребовал бы перечислить ВСЕ кодовые пути каждого репозитория —
    # и молча пропускал бы ревью для расширения, которое забыли внести. Список
    # того, что ревьюировать не надо, ошибается в безопасную сторону: незнакомый
    # файл в него не попадает и ревью получает.
    if config.is_review_pointless(mr.changed_files):
        print(
            f"AI review: пропущено, изменены только {', '.join(mr.changed_files)} — "
            "ни один файл не требует ревью (skip_paths в .ai-review.yml)",
            file=sys.stderr,
        )
        return EXIT_CLEAN

    try:
        claude_md = (repo / "CLAUDE.md").read_text(encoding="utf-8") if (repo / "CLAUDE.md").exists() else ""
        findings = list(run_deterministic(repo, mr.changed_files))
        # Each half of the diff is reviewed by the model its own risk deserves.
        # Routing per merge request instead of per file meant one auth file made
        # the strong model read all nine files of jobbytime-go !29 at $30/M output.
        # Content matters as well as path: `migrations/*` used to send every
        # migration to the strong model, including one whose whole body was
        # ADD COLUMN. See Config.high_risk_content.
        #
        # The file is read from the checkout, not from the diff. A diff would be
        # narrower -- only added lines -- but it would also call a change that
        # DELETES a `DROP COLUMN` line risky, and miss a narrowing statement that
        # was merely moved. Reading the file errs towards the expensive model,
        # which is the direction to err in.
        def read_content(path: str) -> str | None:
            candidate = repo / path
            try:
                if not candidate.is_file():
                    return None
                return candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                # Unreadable is unknown, not safe: is_high_risk falls back to path.
                return None

        high_risk_files, other_files = config.split_by_risk(mr.changed_files, read_content)
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
            findings.extend(
                reviewer.run(system_prompt, build_user_prompt(mr.subset(files)), changed_files=len(files))
            )
    except ReviewIncomplete as exc:
        print(f"AI review: ревью не завершено ({exc}), нужен реран", file=sys.stderr)
        return EXIT_TECHNICAL
    except Exception as exc:  # noqa: BLE001
        print(f"AI review: техническая ошибка ревью: {_describe(exc)}", file=sys.stderr)
        return EXIT_TECHNICAL

    # One line per model finding, before any filtering: a finding dropped below
    # never reaches a thread, and this line is then its only record.
    for finding in findings:
        if finding.reads is not None:
            print(f"AI review: чтения находки fp={finding.fingerprint} [{finding.category}] "
                  f"{finding.file}:{finding.line_start} — {encode_marker(finding.reads)}")

    blocking = keep_blocking(findings)

    try:
        discussions = client.list_discussions(mr_iid)
        decision = decide(blocking, discussions, mr.changed_files, detect_test_globs(repo),
                          mr.added_lines_by_file)
    except Exception as exc:  # noqa: BLE001 - a GitLab failure here is technical, not a verdict
        print(f"AI review: техническая ошибка при работе с GitLab: {_describe(exc)}", file=sys.stderr)
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
            print(f"AI review: не удалось опубликовать находку {finding.fingerprint}: {_describe(exc)}", file=sys.stderr)

    resolved_count = 0
    for discussion_id in decision.to_resolve:
        try:
            client.resolve(mr_iid, discussion_id, "Замечание исправлено, тред закрыт автоматически.")
            resolved_count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"AI review: не удалось закрыть тред {discussion_id}: {_describe(exc)}", file=sys.stderr)

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
            print(f"AI review: не удалось обработать адъюдикацию треда {discussion_id}: {_describe(exc)}", file=sys.stderr)

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
    # And what it was. The drop judges where a finding points, not whether it
    # is right, and the line is the least reliable thing a model names: a true
    # finding anchored one line off looks, without this, exactly like noise.
    for finding in decision.dropped_outside_diff:
        print(f"AI review: отброшенная находка [{finding.category}] "
              f"{finding.file}:{finding.line_start}-{finding.line_end} — "
              f"{_one_line(finding.title)}: {_one_line(finding.problem)}")

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

    code = _run(args)
    # The outcome line is written last and to stderr on purpose. stdout is block
    # buffered when the job log is a pipe rather than a terminal, so a line printed
    # to stdout here can surface above the error it is supposed to summarise.
    sys.stdout.flush()
    print(outcome_line(code), file=sys.stderr)
    return code


def _run(args) -> int:
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
        print(f"AI review: техническая ошибка при старте: {_describe(exc)}", file=sys.stderr)
        return EXIT_TECHNICAL


if __name__ == "__main__":
    raise SystemExit(main())
