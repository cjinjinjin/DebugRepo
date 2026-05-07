"""
Standalone zip image HTTP server.
Replaces: unitorch-service start services/zip_image/config.ini --zip_folder <path>

Serves images from zip files via HTTP.
GET http://0.0.0.0:11230/?file=some_image.jpg

Usage:
    python zip_image_server.py --zip_folder /path/to/zips [--port 11230] [--zip_extension .zip]
"""

import argparse
import os
import zipfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from io import BytesIO


class ZipFileIndex:
    """Index all files inside zip archives in a folder."""

    def __init__(self, zip_folder: str, zip_extension: str = ".zip"):
        self.file_map = {}  # filename -> (zip_path, name_in_zip)
        self.zip_handles = {}  # zip_path -> ZipFile
        self.lock = threading.Lock()
        self._build_index(zip_folder, zip_extension)

    def _build_index(self, zip_folder: str, zip_extension: str):
        if not os.path.isdir(zip_folder):
            print(f"ERROR: zip_folder does not exist: {zip_folder}")
            return

        zip_files = []
        for root, dirs, files in os.walk(zip_folder):
            for f in files:
                if f.endswith(zip_extension):
                    zip_files.append(os.path.join(root, f))

        print(f"Found {len(zip_files)} zip file(s) in {zip_folder}")

        for zp in zip_files:
            try:
                zf = zipfile.ZipFile(zp, "r")
                self.zip_handles[zp] = zf
                for name in zf.namelist():
                    basename = os.path.basename(name)
                    if basename:
                        self.file_map[basename] = (zp, name)
                print(f"  Indexed {zp}: {len(zf.namelist())} entries")
            except Exception as e:
                print(f"  Warning: cannot open {zp}: {e}")

        print(f"Total indexed files: {len(self.file_map)}")

    def read_file(self, filename: str) -> bytes:
        basename = os.path.basename(filename)
        if basename not in self.file_map:
            return None
        zip_path, name_in_zip = self.file_map[basename]
        with self.lock:
            zf = self.zip_handles[zip_path]
            return zf.read(name_in_zip)


# Global index, set after parsing args
_index = None


class ZipImageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        filename = params.get("file", [None])[0]

        if filename is None:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing ?file= parameter")
            return

        data = _index.read_file(filename)
        if data is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(f"File not found: {filename}".encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Suppress per-request logs to reduce noise
        pass


def main():
    global _index
    parser = argparse.ArgumentParser(description="Zip image HTTP server")
    parser.add_argument("--zip_folder", type=str, required=True,
                        help="Directory containing zip files")
    parser.add_argument("--port", type=int, default=11230,
                        help="HTTP port (default: 11230)")
    parser.add_argument("--zip_extension", type=str, default=".zip",
                        help="Zip file extension (default: .zip)")
    args = parser.parse_args()

    _index = ZipFileIndex(args.zip_folder, args.zip_extension)
    server = HTTPServer(("0.0.0.0", args.port), ZipImageHandler)
    print(f"Zip image server running on http://0.0.0.0:{args.port}")
    print(f"  Usage: GET http://0.0.0.0:{args.port}/?file=<filename>")
    server.serve_forever()


if __name__ == "__main__":
    main()
