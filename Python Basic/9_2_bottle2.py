from bottle import route, run, static_file

@route('/')
def main():
    return static_file('.\Python Basic\index.html', root='.')

run(host="localhost", port=9999)