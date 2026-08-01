import pytest

from reviewer.tools import RepoTools, ToolError


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "internal").mkdir()
    (tmp_path / "internal" / "billing.go").write_text(
        "package billing\n\nfunc Charge() error {\n\treturn nil\n}\n", encoding="utf-8"
    )
    (tmp_path / "internal" / "billing_test.go").write_text(
        "package billing\n\nfunc TestCharge(t *testing.T) {}\n", encoding="utf-8"
    )
    (tmp_path / ".env").write_text("STRIPE_SECRET_KEY=do-not-read-me\n", encoding="utf-8")
    (tmp_path / "deploy.pem").write_text("private key\n", encoding="utf-8")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret outside the repo\n", encoding="utf-8")
    return RepoTools(root=tmp_path)


def test_read_file_returns_content(repo):
    content = repo.read_file("internal/billing.go")
    assert "func Charge() error" in content


def test_read_file_respects_offset_and_limit(repo):
    content = repo.read_file("internal/billing.go", offset=3, limit=1)
    assert content.strip() == "func Charge() error {"


def test_read_file_rejects_path_traversal(repo):
    with pytest.raises(ToolError):
        repo.read_file("../outside.txt")


def test_read_file_rejects_absolute_path(repo):
    with pytest.raises(ToolError):
        repo.read_file("/etc/passwd")


@pytest.mark.parametrize("denied", [".env", "deploy.pem"])
def test_read_file_rejects_secret_files(repo, denied):
    with pytest.raises(ToolError):
        repo.read_file(denied)


def test_grep_finds_matches_with_line_numbers(repo):
    hits = repo.grep(r"func Charge", "**/*.go")
    assert hits == ["internal/billing.go:3:func Charge() error {"]


def test_grep_never_returns_denied_files(repo):
    assert repo.grep("STRIPE_SECRET_KEY", "**/*") == []


def test_list_files_excludes_denied(repo):
    listed = repo.list_files("**/*")
    assert "internal/billing.go" in listed
    assert ".env" not in listed
    assert "deploy.pem" not in listed


def test_dispatch_routes_by_name(repo):
    out = repo.dispatch("list_files", {"path_glob": "internal/*.go"})
    assert "internal/billing.go" in out


def test_dispatch_reports_unknown_tool(repo):
    with pytest.raises(ToolError):
        repo.dispatch("write_file", {"path": "a", "content": "b"})


def test_list_files_rejects_traversal_glob(repo):
    listed = repo.list_files("../*")
    assert listed == []


def test_grep_rejects_traversal_glob(repo):
    assert repo.grep("secret outside", "../*") == []


def test_list_files_rejects_absolute_glob(repo):
    with pytest.raises(ToolError):
        repo.list_files("/etc/*")


def test_grep_ignores_symlink_to_denied_file_with_innocuous_name(repo):
    (repo.root / "safe_looking.txt").symlink_to(repo.root / ".env")
    assert repo.grep("STRIPE_SECRET_KEY", "**/*") == []
    assert "safe_looking.txt" not in repo.list_files("**/*")


def test_grep_ignores_symlink_pointing_outside_root(repo):
    outside = repo.root.parent / "outside.txt"
    (repo.root / "escape_link.txt").symlink_to(outside)
    assert repo.grep("secret outside", "**/*") == []
    assert "escape_link.txt" not in repo.list_files("**/*")
