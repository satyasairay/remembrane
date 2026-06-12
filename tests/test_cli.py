import json

from remembrane.cli import main


def test_cli_roundtrip(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    assert main(["--db", db, "store", "the sky is blue", "--importance", "0.9"]) == 0
    assert main(["--db", db, "store", "grass is green"]) == 0
    capsys.readouterr()

    assert main(["--db", db, "recall", "what color is the sky?"]) == 0
    out = capsys.readouterr().out
    assert "sky is blue" in out.splitlines()[0]

    assert main(["--db", db, "stats"]) == 0
    assert "memories: 2" in capsys.readouterr().out

    assert main(["--db", db, "export"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 2


def test_cli_forget(tmp_path, capsys):
    db = str(tmp_path / "cli.db")
    main(["--db", db, "store", "forgettable"])
    out = capsys.readouterr().out
    mem_id = out.split()[-1]
    main(["--db", db, "forget", mem_id])
    assert "forgot 1" in capsys.readouterr().out
