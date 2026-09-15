"""Isolated AKShare full-market fetch bounded by the parent process."""

import sys
from contextlib import redirect_stdout


def main() -> int:
    try:
        with redirect_stdout(sys.stderr):
            import akshare

            frame = akshare.stock_zh_a_spot_em()
        if frame is None or frame.empty:
            return 1
        sys.stdout.write(frame.to_json(orient="split", force_ascii=False))
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
