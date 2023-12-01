import os
import glob

from flask import Flask, render_template, abort, send_file

app = Flask(__name__)

@app.route('/', defaults={'req_path': ''})
@app.route('/<path:req_path>')
def dir_listing(req_path):
    BASE_DIR = os.path.join(os.getcwd(), 'plots')

    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = os.listdir(abs_path)
    no_hidden_files = [file for file in files if file.startswith(".") != True]

    return render_template('files.html', files=no_hidden_files)

def main():
    app.run(debug=True, host='0.0.0.0')

if __name__ == "__main__":
    main()