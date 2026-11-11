"""One JSON request/response per line for the isolated Electron Python process."""
from __future__ import annotations

import contextlib
import json
import sys
import traceback


def main():
    from .service import dispatch

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Запрос должен быть JSON-объектом.")
            request_id = request.get("id")
            with contextlib.redirect_stdout(sys.stderr):
                result = dispatch(request.get("method"), request.get("payload", {}))
            response = {"id": request_id, "result": result}
            encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            encoded = json.dumps({"id": request_id, "error": {"message": str(error)}}, ensure_ascii=False)
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
