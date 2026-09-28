"""Stream one file out of a Kaggle notebook's output, straight to disk.

`kaggle kernels output` reads the whole file into memory before writing it,
which fails with MemoryError on a machine with little free RAM - a 1.2 GB
checkpoint needs 1.2 GB of RAM it does not have. This writes in 8 MB chunks
and prints the sha256 so the copy can be checked against the checksum the
training run registered.

usage:
    python scripts/download_kernel_file.py <owner/kernel> <file in output> <dest>
"""

import hashlib
import os
import sys

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    kernel, wanted, dest = sys.argv[1:4]
    owner, slug = kernel.split("/")

    api = KaggleApi()
    api.authenticate()

    url, token = None, None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            request.page_size = 200
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                if item.file_name == wanted:
                    url = item.url
            token = response.next_page_token
            if url or not token:
                break

    if not url:
        sys.exit(f"{wanted} not found in {kernel} output")

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    digest, written = hashlib.sha256(), 0
    partial = dest + ".part"
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length") or 0)
        with open(partial, "wb") as out:
            for chunk in response.iter_content(8 * 1024 * 1024):
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if total:
                    print(f"\r{written / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    os.replace(partial, dest)
    print(f"\nsaved {dest} ({written:,} bytes)")
    print(f"sha256 {digest.hexdigest()}")


if __name__ == "__main__":
    main()
