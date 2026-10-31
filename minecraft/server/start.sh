#!/bin/sh

set -eu

cd -- "$(dirname -- "$0")"

if ! jar_filename=$(
    python3 -c '
import json
import sys

try:
    with open("paper_manifest.json", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    jar_filename = manifest["jar_filename"]
    if not isinstance(jar_filename, str) or not jar_filename or "/" in jar_filename:
        raise ValueError("jar_filename must be a non-empty filename")
except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
    print("manifest is unreadable: {}".format(error), file=sys.stderr)
    sys.exit(1)

print(jar_filename)
'
); then
    echo "Error: could not resolve the Paper jar from paper_manifest.json." >&2
    exit 1
fi

if [ ! -f "$jar_filename" ]; then
    echo "Error: Paper jar is missing: $jar_filename" >&2
    exit 1
fi

if [ ! -f eula.txt ] || ! grep -Eq '^[[:space:]]*eula=true[[:space:]]*$' eula.txt; then
    echo "EULA not accepted. Review the Minecraft EULA and create $(pwd)/eula.txt containing eula=true yourself before running the server." >&2
    exit 1
fi

exec java -Xms1G -Xmx2G -jar "$jar_filename" nogui
