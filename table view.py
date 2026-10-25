"""
Maintenance utility: delete ALL rows from the parking_logs table.

This is destructive and cannot be undone, so it runs as a dry run unless
--confirm is given. It targets DATABASE_PATH (same default as the app) and
refuses to modify the legacy parking.db in the repository folder.

    python "table view.py"                      # dry run: report how many rows would be deleted
    python "table view.py" --confirm            # delete them
    python "table view.py" --db path/to.db --confirm
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

import config
import database


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description="Delete all rows from parking_logs (dry run by default).")
    parser.add_argument("--db", default=os.environ.get("DATABASE_PATH") or config.DEFAULT_DATABASE_PATH,
                        help="SQLite database path (default: DATABASE_PATH or %(default)s)")
    parser.add_argument("--confirm", action="store_true", help="Actually delete the rows")
    args = parser.parse_args(argv)

    db_path = Path(os.path.abspath(args.db))
    try:
        if not db_path.is_file():
            print(f"Database not found: {db_path}", file=sys.stderr)
            return 1
        # File-identity check: also catches case variants, symlinks and hard links.
        if database.refers_to_committed_baseline(db_path):
            print("Refusing to modify the legacy parking.db in the repository folder. Point --db at a working copy.",
                  file=sys.stderr)
            return 1
    except OSError as e:
        print(f"Could not verify the database path ({type(e).__name__}); nothing was deleted.",
              file=sys.stderr)
        return 1

    try:
        # mode=rw: never create a new database file by accident
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=rw", uri=True)
        try:
            count = conn.execute("SELECT COUNT(*) FROM parking_logs").fetchone()[0]
            if not args.confirm:
                print(f"Dry run: {count} row(s) in parking_logs of {db_path} would be deleted. "
                      f"Re-run with --confirm to delete them.")
                return 0
            conn.execute("DELETE FROM parking_logs")
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as e:
        print(f"Database error ({type(e).__name__}); nothing was deleted.", file=sys.stderr)
        return 1

    print(f"Deleted {count} row(s) from parking_logs in {db_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
