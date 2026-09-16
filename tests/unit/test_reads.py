"""What the reviewer read in a pass, recorded as fact rather than as the model's claim.

On 2026-09-15 seven findings asserted things about the repository that the diff
does not show (go !181, go !186, front !125). The prompt now forbids that (!26),
but whether the model obeys could not be measured: the log only had a byte total.
These tests pin the record that makes the measurement possible."""

from reviewer.reads import Read, decode_marker, encode_marker, reads_from_call


def test_read_file_records_the_lines_actually_returned_not_the_ones_asked_for():
    result = "line 10\nline 11\nline 12"
    reads = reads_from_call("read_file", {"path": "a.go", "offset": 10, "limit": 200}, result)
    assert reads == [Read("read_file", "a.go", 10, 12)]


def test_read_file_defaults_to_the_first_line():
    reads = reads_from_call("read_file", {"path": "a.go"}, "x\ny")
    assert reads == [Read("read_file", "a.go", 1, 2)]


def test_read_file_past_the_end_saw_nothing_and_is_not_a_read():
    assert reads_from_call("read_file", {"path": "a.go", "offset": 900}, "") == []


def test_grep_records_each_hit_as_a_seen_line_of_its_file():
    result = "internal/x.go:12:func A()\ninternal/x.go:15:func B()\ny.yml:3:stage: test"
    reads = reads_from_call("grep", {"pattern": "func"}, result)
    assert reads == [
        Read("grep", "internal/x.go", 12, 12),
        Read("grep", "internal/x.go", 15, 15),
        Read("grep", "y.yml", 3, 3),
    ]


def test_grep_hit_whose_text_contains_colons_keeps_the_path_intact():
    reads = reads_from_call("grep", {"pattern": "a"}, "a.yml:7:image: x:1:2")
    assert reads == [Read("grep", "a.yml", 7, 7)]


def test_grep_with_no_hits_read_nothing():
    assert reads_from_call("grep", {"pattern": "zzz"}, "") == []


def test_list_files_records_the_mask_not_the_files():
    # Listing proves a file exists, not what is in it: it must not count as reading
    # any of the files it names.
    reads = reads_from_call("list_files", {"path_glob": "*_test.go"}, "a_test.go\nb_test.go")
    assert reads == [Read("list_files", "*_test.go", None, None)]


def test_list_files_without_a_mask_records_the_default_one():
    assert reads_from_call("list_files", {}, "a.go") == [Read("list_files", "**/*", None, None)]


def test_unknown_tool_reads_nothing():
    assert reads_from_call("submit_findings", {}, "x") == []


def test_marker_groups_lines_per_file_and_merges_adjacent_ranges():
    marker = encode_marker([
        Read("read_file", "a.go", 1, 200),
        Read("read_file", "a.go", 201, 250),
        Read("read_file", "a.go", 400, 410),
        Read("grep", "b.go", 12, 12),
        Read("grep", "b.go", 13, 13),
        Read("grep", "b.go", 40, 40),
        Read("list_files", "**/*_test.go", None, None),
    ])
    assert marker == (
        "v=1;read=a.go:1-250,400-410;grep=b.go:12-13,40;list=%2A%2A/%2A_test.go"
    )


def test_marker_of_a_pass_that_read_nothing_says_so_explicitly():
    # An empty record is the most important one to keep: it is a finding formed
    # without reading anything.
    assert encode_marker([]) == "v=1;read=;grep=;list="


def test_marker_round_trips_paths_with_separator_characters():
    reads = [
        Read("read_file", "dir with space/a;b|c,d:e=f.go", 3, 4),
        Read("grep", "ci/.gitlab-ci.yml", 1100, 1100),
        Read("list_files", "deploy/*.yml", None, None),
    ]
    decoded = decode_marker(encode_marker(reads))
    assert decoded == {
        "read": {"dir with space/a;b|c,d:e=f.go": [(3, 4)]},
        "grep": {"ci/.gitlab-ci.yml": [(1100, 1100)]},
        "list": ["deploy/*.yml"],
    }


def test_marker_can_never_close_the_html_comment_it_lives_in():
    marker = encode_marker([Read("read_file", "evil-->x.go", 1, 1), Read("list_files", "--><b>", None, None)])
    assert "-->" not in marker
    assert ">" not in marker
