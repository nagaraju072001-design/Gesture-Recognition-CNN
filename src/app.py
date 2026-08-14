from flask import Flask, Response, render_template, jsonify

from stream import VideoStream

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


video_stream = VideoStream()


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route("/video_feed")
def video_feed():

    return Response(
        video_stream.generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.route("/api/status")
def api_status():

    return jsonify(
        video_stream.get_status()
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
