import pytest

from reviewer.config import Limits
from reviewer.diff import (
    DiffTooLarge,
    FileDiff,
    MergeRequestDiff,
    check_limits,
    parse_new_lines,
    render_diff_block,
)

SAMPLE_DIFF = """@@ -10,6 +10,9 @@ func Charge() error {
 	client := stripe.New()
-	resp, err := client.Charge(ctx)
+	resp, err := client.Charge(ctx)
+	if err != nil {
+		return err
+	}
 	return nil
 }
"""


def make_mr(files):
    return MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40, files=tuple(files)
    )


def test_parse_new_lines_returns_added_line_numbers():
    assert parse_new_lines(SAMPLE_DIFF) == (11, 12, 13, 14)


def test_parse_new_lines_handles_multiple_hunks():
    diff = "@@ -1,2 +1,2 @@\n-a\n+A\n@@ -20,2 +30,3 @@\n b\n+C\n"
    assert parse_new_lines(diff) == (1, 31)


def test_changed_line_count_sums_files():
    mr = make_mr([
        FileDiff("a.go", "a.go", SAMPLE_DIFF, parse_new_lines(SAMPLE_DIFF)),
        FileDiff("b.go", "b.go", "@@ -1 +1,2 @@\n x\n+y\n", (2,)),
    ])
    assert mr.changed_line_count == 5
    assert mr.changed_files == ("a.go", "b.go")


def test_check_limits_passes_within_bounds():
    mr = make_mr([FileDiff("a.go", "a.go", SAMPLE_DIFF, parse_new_lines(SAMPLE_DIFF))])
    check_limits(mr, Limits(max_files=10, max_changed_lines=10))


def test_check_limits_rejects_too_many_files():
    files = [FileDiff(f"f{i}.go", f"f{i}.go", "@@ -1 +1 @@\n+x\n", (1,)) for i in range(5)]
    with pytest.raises(DiffTooLarge) as exc:
        check_limits(make_mr(files), Limits(max_files=3, max_changed_lines=100))
    assert "файл" in str(exc.value)


def test_check_limits_rejects_too_many_lines():
    mr = make_mr([FileDiff("a.go", "a.go", SAMPLE_DIFF, parse_new_lines(SAMPLE_DIFF))])
    with pytest.raises(DiffTooLarge):
        check_limits(mr, Limits(max_files=10, max_changed_lines=2))


def test_render_diff_block_marks_files_and_is_stable():
    mr = make_mr([FileDiff("a.go", "a.go", SAMPLE_DIFF, parse_new_lines(SAMPLE_DIFF))])
    block = render_diff_block(mr)
    assert block.startswith("### a.go")
    assert render_diff_block(mr) == block
