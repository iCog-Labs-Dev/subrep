from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import urllib.request


def fail(message):
    print("Error: {}".format(message), file=sys.stderr)
    return 1


def load_manifest(manifest_path):
    try:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except FileNotFoundError:
        return None, "manifest not found at {}".format(manifest_path)
    except (OSError, ValueError) as error:
        return None, "manifest is malformed: {}".format(error)

    required_fields = ("jar_filename", "download_url", "sha256")
    if not isinstance(manifest, dict):
        return None, "manifest is malformed: expected a JSON object"
    if any(field not in manifest for field in required_fields):
        return None, "manifest is malformed: missing a required download field"
    if not all(isinstance(manifest[field], str) for field in required_fields):
        return None, "manifest is malformed: download fields must be strings"
    if len(manifest["sha256"]) != 64:
        return None, "manifest is malformed: SHA-256 must contain 64 characters"
    return manifest, None


def sha256_for(path):
    digest = hashlib.sha256()
    with path.open("rb") as jar_file:
        for chunk in iter(lambda: jar_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_eula(server_directory):
    eula_path = server_directory / "eula.txt"
    try:
        accepted = any(
            line.strip().lower() == "eula=true"
            for line in eula_path.read_text(encoding="utf-8").splitlines()
        )
    except (FileNotFoundError, OSError, UnicodeError):
        accepted = False
    if not accepted:
        print(
            "EULA not accepted. Review the Minecraft EULA and create "
            "{} containing eula=true yourself before running the server.".format(
                eula_path
            )
        )


def main():
    server_directory = Path(__file__).resolve().parent
    manifest_path = server_directory / "paper_manifest.json"
    manifest, error = load_manifest(manifest_path)
    if error is not None:
        return fail(error)

    jar_path = server_directory / manifest["jar_filename"]
    expected_hash = manifest["sha256"].lower()
    if jar_path.is_file():
        try:
            actual_hash = sha256_for(jar_path)
        except OSError as error:
            return fail("could not read existing jar: {}".format(error))
        if actual_hash == expected_hash:
            print(
                "{} is already present and SHA-256 verified: {}".format(
                    jar_path.name, actual_hash
                )
            )
            report_eula(server_directory)
            return 0
        try:
            jar_path.unlink()
        except OSError as error:
            return fail("existing jar has the wrong SHA-256 and could not be removed: {}".format(error))

    temporary_path = server_directory / (jar_path.name + ".part")
    try:
        if temporary_path.exists():
            temporary_path.unlink()
        request = urllib.request.Request(
            manifest["download_url"], headers={"User-Agent": "subrep-paper-downloader"}
        )
        with urllib.request.urlopen(request) as response:
            with temporary_path.open("wb") as temporary_file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary_file.write(chunk)
        actual_hash = sha256_for(temporary_path)
        if actual_hash != expected_hash:
            temporary_path.unlink()
            return fail(
                "downloaded jar SHA-256 mismatch: expected {}, got {}".format(
                    expected_hash, actual_hash
                )
            )
        temporary_path.replace(jar_path)
    except (OSError, ValueError) as error:
        try:
            if temporary_path.exists():
                temporary_path.unlink()
        except OSError:
            pass
        return fail("could not download Paper: {}".format(error))

    print("{} downloaded and SHA-256 verified: {}".format(jar_path.name, actual_hash))
    report_eula(server_directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
