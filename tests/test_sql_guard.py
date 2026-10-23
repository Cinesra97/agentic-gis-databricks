import pytest

from agentic_gis.security.sql_guard import SQLGuard, SQLPolicy

LOCAL = SQLPolicy(
    "duckdb", {"demo_311_sanitation": "demo_311_sanitation", "nyc_boroughs": "nyc_boroughs"}, 1000
)
DBX = SQLPolicy("databricks", {"cat.sch.sr311": "dbx_311"}, 500, {"dbx_311": {"caller_phone"}}, "cat", "sch")
guard = SQLGuard()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT complaint_type, COUNT(*) AS n FROM demo_311_sanitation GROUP BY 1 ORDER BY n DESC",
        "WITH c AS (SELECT * FROM demo_311_sanitation) SELECT borough, COUNT(*) FROM c GROUP BY 1",
        "SELECT b.boro_name, COUNT(*) FROM demo_311_sanitation c JOIN nyc_boroughs b "
        "ON ST_Within(c.geometry, b.geometry) GROUP BY 1",
        "SELECT * FROM main.demo_311_sanitation",
        "SELECT borough FROM demo_311_sanitation UNION ALL SELECT borough FROM nyc_boroughs",
        "(SELECT 1 AS x FROM demo_311_sanitation)",
    ],
)
def test_allows_read_only_queries(sql):
    r = guard.validate(sql, LOCAL)
    assert r.ok, r.errors
    assert "LIMIT" in r.sql.upper()
    assert r.fingerprint


@pytest.mark.parametrize(
    "sql,needle",
    [
        ("DROP TABLE demo_311_sanitation", "DROP"),
        ("DELETE FROM demo_311_sanitation", "DELETE"),
        ("INSERT INTO demo_311_sanitation SELECT * FROM demo_311_sanitation", "INSERT"),
        ("UPDATE demo_311_sanitation SET borough = 'X'", "UPDATE"),
        ("CREATE TABLE x AS SELECT 1", "CREATE"),
        ("SELECT * FROM demo_311_sanitation; DROP TABLE x", "Exactly one"),
        ("SELECT * FROM read_csv('/etc/passwd')", "read_csv"),
        ("SELECT * FROM 'secrets.csv'", "File paths"),
        ("SELECT getenv('HOME')", "getenv"),
        ("COPY demo_311_sanitation TO 'out.csv'", "COPY"),
        ("ATTACH 'evil.db'", "ATTACH"),
        ("INSTALL httpfs", "INSTALL"),
        ("PRAGMA show_tables", "PRAGMA"),
        ("SET enable_external_access = true", "SET"),
        ("SELECT * FROM other_table", "allowlist"),
        ("SELECT * FROM duckdb_settings()", "Table functions"),
        ("SELECT 1 INTO t2 FROM demo_311_sanitation", "INTO"),
        ("SELECT * FROM (SELECT 1) t, LATERAL (SELECT * FROM read_parquet('x'))", "read_parquet"),
        ("", "Empty"),
        ("SELEC * FROM", "parsed"),
    ],
)
def test_blocks_unsafe_sql(sql, needle):
    r = guard.validate(sql, LOCAL)
    assert not r.ok
    assert any(needle.lower() in e.lower() for e in r.errors), r.errors


def test_limit_enforced_and_reduced():
    r = guard.validate("SELECT * FROM demo_311_sanitation LIMIT 999999", LOCAL)
    assert r.ok and r.limit == 1000 and r.sql.endswith("LIMIT 1000")
    assert any("reduced" in w for w in r.warnings)
    r2 = guard.validate("SELECT * FROM demo_311_sanitation LIMIT 10", LOCAL)
    assert r2.limit == 10


def test_comments_removed_and_fingerprint_stable():
    a = guard.validate("SELECT borough FROM demo_311_sanitation -- hi; DROP TABLE x", LOCAL)
    b = guard.validate("select borough from demo_311_sanitation", LOCAL)
    assert a.ok and "--" not in a.sql
    assert a.fingerprint == b.fingerprint


def test_databricks_qualification_and_policies():
    r = guard.validate("SELECT complaint_type FROM sr311", DBX)
    assert r.ok
    assert "cat.sch.sr311" in r.sql.lower()
    assert r.datasets == ["dbx_311"]
    for sql, needle in [
        ("SELECT caller_phone FROM cat.sch.sr311", "sensitive"),
        ("SELECT * FROM cat.sch.sr311", "SELECT *"),
        ("SELECT secret('scope', 'key') FROM sr311", "secret"),
        ("SELECT ai_query('m', complaint_type) FROM sr311", "ai_query"),
        ("SELECT * FROM IDENTIFIER(:t)", "Table functions"),
        ("SELECT cat.sch.my_udf(complaint_type) FROM sr311", "user-defined"),
        ("SELECT * FROM delta.`/mnt/raw`", "File paths"),
        ("SELECT * FROM system.access.audit", "allowlist"),
        ("SELECT http_request('x') FROM sr311", "http_request"),
    ]:
        res = guard.validate(sql, DBX)
        assert not res.ok, sql
        assert any(needle.lower() in e.lower() for e in res.errors), (sql, res.errors)
