import os
import uuid
import subprocess
import tempfile
import logging
import requests

logger = logging.getLogger(__name__)


class VideoManager:
    def __init__(self, use_file_buffer: bool = False, temp_dir: str | None = None):
        """
        Args:
            use_file_buffer (bool):
                - If True: methods that fetch video will return bytes.
                - If False: methods will return a filesystem path.
            temp_dir (str|None): Directory in which to place temp files.
                Defaults to the system temp directory.
        """
        self.use_file_buffer = use_file_buffer
        self.temp_dir = temp_dir or tempfile.gettempdir()

    def _read_hls_stream_to_buffer(
        self, m3u8_url: str, output_path: str | None = None
    ) -> bytes | str:
        """
        Reads an HLS (.m3u8) stream via FFmpeg into a temp MP4 (or
        `output_path` if provided), then returns its bytes or the file path.
        """
        # Determine where to write the MP4
        if output_path:
            tmp_path = output_path
        else:
            # Create a true NamedTemporaryFile for the MP4 output
            with tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False, dir=self.temp_dir
            ) as tmp_mp4:
                tmp_path = tmp_mp4.name

        command = [
            "ffmpeg",
            "-i",
            m3u8_url,
            "-c",
            "copy",
            "-bsf:a",
            "aac_adtstoasc",
            "-f",
            "mp4",
            tmp_path,
            "-movflags",
            "frag_keyframe+empty_moov",
        ]

        try:
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg error: {result.stderr.decode(errors='ignore')}"
                )

            if self.use_file_buffer:
                with open(tmp_path, "rb") as f:
                    data = f.read()
                return data
            else:
                return tmp_path

        except Exception as e:
            logger.error(f"Error reading HLS stream: {e}")
            # Clean up on error if we created the file
            if not output_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _download_video(self, video_url: str, output_path: str | None = None) -> str:
        """
        Downloads a non-HLS video (e.g. .mp4) into `output_path` if given,
        otherwise into a temp file, and returns its path.
        """
        if output_path:
            out_path = output_path
        else:
            _, ext = os.path.splitext(video_url)
            local_name = f"video_{uuid.uuid4()}{ext or '.mp4'}"
            out_path = os.path.join(self.temp_dir, local_name)

        resp = requests.get(video_url, stream=True)
        resp.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        logger.info(f"Video downloaded to {out_path}")
        return out_path

    def _read_video_to_buffer(self, video_url: str) -> bytes:
        """
        Streams a non-HLS video URL directly into memory and returns its bytes.
        """
        resp = requests.get(video_url, stream=True)
        resp.raise_for_status()
        return resp.content

    def get_video_data(
        self, video_url: str, output_video_path: str | None = None
    ) -> bytes | str:
        """
        High‐level dispatcher:
          - If HLS (.m3u8), use FFmpeg → MP4, writing to `output_video_path` if given.
          - Else if .mp4 (or other), either download to `output_video_path` or buffer in RAM.
        """
        url = video_url.lower()

        if url.endswith(".m3u8"):
            return self._read_hls_stream_to_buffer(
                m3u8_url=video_url, output_path=output_video_path
            )

        # non-HLS case
        if self.use_file_buffer:
            return self._read_video_to_buffer(video_url)
        else:
            return self._download_video(
                video_url=video_url, output_path=output_video_path
            )
